import argparse
import functools
import itertools
import sys
from abc import ABC
from typing import Optional, List, TextIO, Union, Dict, Iterable, Callable, Tuple

from llvmlite import binding, ir  # type: ignore


def ir_const_int(v: int, width: int) -> ir.Constant:
    return ir.Constant(ir.IntType(width), v)


class InputGenerator:
    builder: ir.IRBuilder
    total_used: int

    def __init__(self, builder: ir.IRBuilder):
        self.builder = builder
        self.total_used = 0

    def generate_input(self, width: int) -> ir.Value:
        raise NotImplementedError


class RandInputGenerator(InputGenerator):
    rand_function: ir.Function

    def __init__(self, builder: ir.IRBuilder, rand_function: ir.Function):
        super().__init__(builder)
        self.rand_function = rand_function

    def generate_input(self, width: int) -> ir.Value:
        if width < 32:
            return self.builder.trunc(self.builder.call(self.rand_function, ()), ir.IntType(width))

        n: int = int((width + 31) / 32)
        self.total_used += n
        p: ir.Value = self.builder.alloca(ir.IntType(32), n)
        for i in range(n):
            self.builder.store(self.builder.call(self.rand_function, ()),
                               self.builder.gep(p, (ir.Constant(ir.IntType(32), i),)))

        return self.builder.load(self.builder.bitcast(p, ir.IntType(width).as_pointer()))


class FuzzerInputGenerator(InputGenerator):
    data: ir.Value
    used: ir.Value

    def __init__(self, builder: ir.IRBuilder, data: ir.Value, used: ir.Value):
        super().__init__(builder)
        self.builder = builder
        self.data = data
        self.used = used

    def generate_input(self, width: int) -> ir.Value:
        n: int = int((width + 7) / 8)
        self.total_used += n

        r: ir.Value = self.builder.trunc(self.builder.load(self.builder.bitcast(
            self.builder.gep(self.data, (self.used,)), ir.IntType(n * 8).as_pointer())),
            ir.IntType(width))
        self.used = self.builder.add(self.used, ir_const_int(n, 64))
        return r


class Node(ABC):
    nid: int
    symbol: str

    def __init__(self, nid: int, symbol: str = ""):
        self.nid = nid
        self.symbol = symbol


class Sort(Node):
    @property
    def sid(self) -> int:
        return self.nid

    def to_ir_type(self) -> ir.Type:
        raise NotImplementedError


class BitvecSort(Sort):
    width: int

    def __init__(self, nid: int, width: int, symbol: str = ""):
        super().__init__(nid, symbol)
        self.width = width

    def to_ir_type(self) -> ir.IntType:
        return ir.IntType(self.width)


class ArraySort(Sort):
    index_sort: Sort
    element_sort: Sort

    def __init__(self, nid: int, index_sort: Sort, element_sort: Sort, symbol: str = ""):
        super().__init__(nid, symbol)
        self.index_sort = index_sort
        self.element_sort = element_sort

    def to_ir_type(self) -> ir.ArrayType:
        # if isinstance(self.index_sort, BitvecSort):
        #     return ir.ArrayType(self.element_sort.to_ir_type(), 1 << self.index_sort.width)
        # TODO: support array as index
        raise NotImplementedError


class Expr(Node):
    sort: Sort

    def __init__(self, nid: int, sort: Sort, symbol: str = ""):
        super().__init__(nid, symbol)
        self.sort = sort

    def get_child_exprs(self) -> Iterable['Expr']:
        raise NotImplementedError

    def get_leaf_exprs(self) -> Iterable['Expr']:
        if self.get_child_exprs():
            return (expr for child_expr in self.get_child_exprs()
                    for expr in child_expr.get_leaf_exprs())
        return self,

    def to_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        if self.nid in m:
            return m[self.nid]
        v: ir.Value = self.to_new_ir_value(builder, m)
        m[self.nid] = v
        return v

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise NotImplementedError


class Bitvec(Expr, ABC):
    sort: BitvecSort


class Array(Expr, ABC):
    sort: ArraySort


class Minus(Bitvec):
    bitvec: Bitvec

    def __init__(self, bitvec: Bitvec):
        super().__init__(-bitvec.nid, bitvec.sort)
        self.bitvec = bitvec

    def get_child_exprs(self) -> Iterable['Expr']:
        return self.bitvec,

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.not_(self.bitvec.to_ir_value(builder, m))


class Input(Expr, ABC):
    def get_child_exprs(self) -> Iterable['Expr']:
        return ()


class BitvecInput(Bitvec, Input):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise ValueError


class ArrayInput(Array, Input):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise ValueError


class Constant(Bitvec):
    value: int

    def __init__(self, nid: int, sort: BitvecSort, value: int, symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.value = value

    def get_child_exprs(self) -> Iterable['Expr']:
        return ()

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return ir.Constant(self.sort.to_ir_type(), self.value)


class One(Constant):
    def __init__(self, nid: int, sort: BitvecSort, symbol: str = ""):
        super().__init__(nid, sort, 1, symbol)


class Ones(Constant):
    def __init__(self, nid: int, sort: BitvecSort, symbol: str = ""):
        super().__init__(nid, sort, -1, symbol)


class Zero(Constant):
    def __init__(self, nid: int, sort: BitvecSort, symbol: str = ""):
        super().__init__(nid, sort, 0, symbol)


class Const(Constant):
    def __init__(self, nid: int, sort: BitvecSort, bin_str: str, symbol: str = ""):
        super().__init__(nid, sort, int(bin_str, 2), symbol)


class Constd(Constant):
    def __init__(self, nid: int, sort: BitvecSort, dec_str: str, symbol: str = ""):
        super().__init__(nid, sort, int(dec_str, 10), symbol)


class Consth(Constant):
    def __init__(self, nid: int, sort: BitvecSort, hex_str: str, symbol: str = ""):
        super().__init__(nid, sort, int(hex_str, 16), symbol)


class State(Expr, ABC):
    init: Optional[Expr]
    next: Optional[Expr]

    def __init__(self, nid: int, sort: Sort, symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.init = None
        self.next = None

    def get_child_exprs(self) -> Iterable['Expr']:
        return ()


class BitvecState(Bitvec, State):
    init: Optional[Bitvec]
    next: Optional[Bitvec]

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise ValueError


class ArrayState(Array, State):
    next: Optional[Array]

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise ValueError


class Ext(Bitvec, ABC):
    bitvec: Bitvec
    w: int

    def __init__(self, nid: int, sort: BitvecSort, bitvec: Bitvec, w: int, symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.bitvec = bitvec
        self.w = w

    def get_child_exprs(self) -> Iterable['Expr']:
        return self.bitvec,


class Sext(Ext):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.sext(self.bitvec.to_ir_value(builder, m), self.sort.to_ir_type())


class Uext(Ext):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.zext(self.bitvec.to_ir_value(builder, m), self.sort.to_ir_type())


class Slice(Bitvec):
    bitvec: Bitvec
    upper: int
    lower: int

    def __init__(self, nid: int, sort: BitvecSort, bitvec: Bitvec, upper: int, lower: int,
                 symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.bitvec = bitvec
        self.upper = upper
        self.lower = lower

    def get_child_exprs(self) -> Iterable['Expr']:
        return self.bitvec,

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.trunc(
            builder.lshr(self.bitvec.to_ir_value(builder, m),
                         ir.Constant(self.bitvec.sort.to_ir_type(), self.lower)),
            self.sort.to_ir_type())


class BitvecUnaryOp(Bitvec, ABC):
    bitvec: Bitvec

    def __init__(self, nid: int, sort: BitvecSort, bitvec: Bitvec, symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.bitvec = bitvec

    def get_child_exprs(self) -> Iterable['Expr']:
        return self.bitvec,


class Not(BitvecUnaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.not_(self.bitvec.to_ir_value(builder, m))


class Inc(BitvecUnaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.add(self.bitvec.to_ir_value(builder, m),
                           ir.Constant(self.sort.to_ir_type(), 1))


class Dec(BitvecUnaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.sub(self.bitvec.to_ir_value(builder, m),
                           ir.Constant(self.sort.to_ir_type(), 1))


class Neg(BitvecUnaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.neg(self.bitvec.to_ir_value(builder, m))


class Redand(BitvecUnaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.icmp_signed('==', self.bitvec.to_ir_value(builder, m),
                                   ir.Constant(self.bitvec.sort.to_ir_type(), -1))


class Redor(BitvecUnaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.icmp_signed('!=', self.bitvec.to_ir_value(builder, m),
                                   ir.Constant(self.bitvec.sort.to_ir_type(), 0))


class Redxor(BitvecUnaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise NotImplementedError


class BitvecBinaryOp(Bitvec, ABC):
    bitvec1: Bitvec
    bitvec2: Bitvec

    def __init__(self, nid: int, sort: BitvecSort, bitvec1: Bitvec, bitvec2: Bitvec,
                 symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.bitvec1 = bitvec1
        self.bitvec2 = bitvec2

    def get_child_exprs(self) -> Iterable['Expr']:
        return self.bitvec1, self.bitvec2


class Iff(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.icmp_signed('==', self.bitvec1.to_ir_value(builder, m),
                                   self.bitvec2.to_ir_value(builder, m))


class Implies(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.or_(builder.not_(self.bitvec1.to_ir_value(builder, m)),
                           self.bitvec2.to_ir_value(builder, m))


class Equality(Bitvec, ABC):
    expr1: Expr
    expr2: Expr

    def __init__(self, nid: int, sort: BitvecSort, expr1: Expr, expr2, symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.expr1 = expr1
        self.expr2 = expr2

    def get_child_exprs(self) -> Iterable['Expr']:
        return self.expr1, self.expr2


class Eq(Equality):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        if not isinstance(self.expr1, Bitvec):
            raise NotImplementedError
        return builder.icmp_signed('==', self.expr1.to_ir_value(builder, m),
                                   self.expr2.to_ir_value(builder, m))


class Neq(Equality):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        if not isinstance(self.expr1, Bitvec):
            raise NotImplementedError
        return builder.icmp_signed('!=', self.expr1.to_ir_value(builder, m),
                                   self.expr2.to_ir_value(builder, m))


class Sgt(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.icmp_signed('>', self.bitvec1.to_ir_value(builder, m),
                                   self.bitvec2.to_ir_value(builder, m))


class Ugt(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.icmp_unsigned('>', self.bitvec1.to_ir_value(builder, m),
                                     self.bitvec2.to_ir_value(builder, m))


class Sgte(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.icmp_signed('>=', self.bitvec1.to_ir_value(builder, m),
                                   self.bitvec2.to_ir_value(builder, m))


class Ugte(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.icmp_unsigned('>=', self.bitvec1.to_ir_value(builder, m),
                                     self.bitvec2.to_ir_value(builder, m))


class Slt(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.icmp_signed('<', self.bitvec1.to_ir_value(builder, m),
                                   self.bitvec2.to_ir_value(builder, m))


class Ult(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.icmp_unsigned('<', self.bitvec1.to_ir_value(builder, m),
                                     self.bitvec2.to_ir_value(builder, m))


class Slte(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.icmp_signed('<=', self.bitvec1.to_ir_value(builder, m),
                                   self.bitvec2.to_ir_value(builder, m))


class Ulte(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.icmp_unsigned('<=', self.bitvec1.to_ir_value(builder, m),
                                     self.bitvec2.to_ir_value(builder, m))


class And(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.and_(self.bitvec1.to_ir_value(builder, m),
                            self.bitvec2.to_ir_value(builder, m))


class Nand(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.not_(builder.and_(self.bitvec1.to_ir_value(builder, m),
                                         self.bitvec2.to_ir_value(builder, m)))


class Nor(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.not_(builder.or_(self.bitvec1.to_ir_value(builder, m),
                                        self.bitvec2.to_ir_value(builder, m)))


class Or(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.or_(self.bitvec1.to_ir_value(builder, m),
                           self.bitvec2.to_ir_value(builder, m))


class Xnor(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.not_(builder.xor(self.bitvec1.to_ir_value(builder, m),
                                        self.bitvec2.to_ir_value(builder, m)))


class Xor(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.xor(self.bitvec1.to_ir_value(builder, m),
                           self.bitvec2.to_ir_value(builder, m))


class Sll(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.shl(self.bitvec1.to_ir_value(builder, m),
                           self.bitvec2.to_ir_value(builder, m))


class Sra(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.ashr(self.bitvec1.to_ir_value(builder, m),
                            self.bitvec2.to_ir_value(builder, m))


class Srl(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.lshr(self.bitvec1.to_ir_value(builder, m),
                            self.bitvec2.to_ir_value(builder, m))


class Rol(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise NotImplementedError


class Ror(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise NotImplementedError


class Add(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.add(self.bitvec1.to_ir_value(builder, m),
                           self.bitvec2.to_ir_value(builder, m))


class Mul(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.mul(self.bitvec1.to_ir_value(builder, m),
                           self.bitvec2.to_ir_value(builder, m))


class Sdiv(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.sdiv(self.bitvec1.to_ir_value(builder, m),
                            self.bitvec2.to_ir_value(builder, m))


class Udiv(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.udiv(self.bitvec1.to_ir_value(builder, m),
                            self.bitvec2.to_ir_value(builder, m))


class Smod(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.module(self.bitvec1.to_ir_value(builder, m),
                              self.bitvec2.to_ir_value(builder, m))


class Srem(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.srem(self.bitvec1.to_ir_value(builder, m),
                            self.bitvec2.to_ir_value(builder, m))


class Urem(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.urem(self.bitvec1.to_ir_value(builder, m),
                            self.bitvec2.to_ir_value(builder, m))


class Sub(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.sub(self.bitvec1.to_ir_value(builder, m),
                           self.bitvec2.to_ir_value(builder, m))


class Saddo(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.extract_value(
            builder.sadd_with_overflow(self.bitvec1.to_ir_value(builder, m),
                                       self.bitvec2.to_ir_value(builder, m)), 1)


class Uaddo(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.extract_value(
            builder.uadd_with_overflow(self.bitvec1.to_ir_value(builder, m),
                                       self.bitvec2.to_ir_value(builder, m)), 1)


class Sdivo(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        t: ir.IntType = self.bitvec1.sort.to_ir_type()
        return builder.and_(
            builder.icmp_signed('==', self.bitvec1.to_ir_value(builder, m),
                                ir.Constant(t, 1 << self.bitvec1.sort.width - 1)),
            builder.icmp_signed('==', self.bitvec2.to_ir_value(builder, m),
                                ir.Constant(t, -1)))


class Udivo(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return ir.Constant(ir.IntType(1), False)


class Smulo(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.extract_value(
            builder.smul_with_overflow(self.bitvec1.to_ir_value(builder, m),
                                       self.bitvec2.to_ir_value(builder, m)), 1)


class Umulo(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.extract_value(
            builder.umul_with_overflow(self.bitvec1.to_ir_value(builder, m),
                                       self.bitvec2.to_ir_value(builder, m)), 1)


class Ssubo(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.extract_value(
            builder.ssub_with_overflow(self.bitvec1.to_ir_value(builder, m),
                                       self.bitvec2.to_ir_value(builder, m)), 1)


class Usubo(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.extract_value(
            builder.usub_with_overflow(self.bitvec1.to_ir_value(builder, m),
                                       self.bitvec2.to_ir_value(builder, m)), 1)


class Concat(BitvecBinaryOp):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        t: ir.IntType = self.sort.to_ir_type()
        return builder.or_(
            builder.shl(builder.zext(self.bitvec1.to_ir_value(builder, m), t),
                        ir.Constant(t, self.bitvec2.sort.width)),
            builder.zext(self.bitvec2.to_ir_value(builder, m), t))


class Read(Expr, ABC):
    array: Array
    index_expr: Expr

    def __init__(self, nid: int, sort: Sort, array: Array, index_expr: Expr, symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.array = array
        self.index_expr = index_expr

    def get_child_exprs(self) -> Iterable['Expr']:
        return self.array, self.index_expr


class BitvecRead(Bitvec, Read):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise NotImplementedError


class ArrayRead(Array, Read):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise NotImplementedError


class Ite(Expr, ABC):
    cond_bitvec: Bitvec
    then_expr: Expr
    else_expr: Expr

    def __init__(self, nid: int, sort: Sort, cond_bitvec: Bitvec, then_expr: Expr, else_expr: Expr,
                 symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.cond_bitvec = cond_bitvec
        self.then_expr = then_expr
        self.else_expr = else_expr

    def get_child_exprs(self) -> Iterable['Expr']:
        return self.cond_bitvec, self.then_expr, self.else_expr


class ArrayIte(Array, Ite):
    then_expr: Array
    else_expr: Array

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise NotImplementedError


class BitvecIte(Bitvec, Ite):
    then_expr: Bitvec
    else_expr: Bitvec

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.select(self.cond_bitvec.to_ir_value(builder, m),
                              self.then_expr.to_ir_value(builder, m),
                              self.else_expr.to_ir_value(builder, m))


class Write(Array):
    array: Array
    index_expr: Expr
    element_expr: Expr

    def __init__(self, nid: int, sort: ArraySort, array: Array, index_expr: Expr,
                 element_expr: Expr, symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.array = array
        self.index_expr = index_expr
        self.element_expr = element_expr

    def get_child_exprs(self) -> Iterable['Expr']:
        return self.array, self.index_expr, self.element_expr

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise NotImplementedError


class BitvecInit(Node):
    sort: BitvecSort
    state: BitvecState
    bitvec: Bitvec

    def __init__(self, nid: int, sort: BitvecSort, state: BitvecState, bitvec: Bitvec,
                 symbol: str = ""):
        super().__init__(nid, symbol)
        self.sort = sort
        self.state = state
        self.bitvec = bitvec


class ArrayInit(Node):
    sort: ArraySort
    state: ArrayState
    expr: Expr

    def __init__(self, nid: int, sort: ArraySort, state: ArrayState, expr: Expr,
                 symbol: str = ""):
        super().__init__(nid, symbol)
        self.sort = sort
        self.state = state
        self.expr = expr


class BitvecNext(Node):
    sort: BitvecSort
    state: BitvecState
    bitvec: Bitvec

    def __init__(self, nid: int, sort: BitvecSort, state: BitvecState, bitvec: Bitvec,
                 symbol: str = ""):
        super().__init__(nid, symbol)
        self.sort = sort
        self.state = state
        self.bitvec = bitvec


class ArrayNext(Node):
    sort: ArraySort
    state: ArrayState
    array: Array

    def __init__(self, nid: int, sort: ArraySort, state: ArrayState, array: Array,
                 symbol: str = ""):
        super().__init__(nid, symbol)
        self.sort = sort
        self.state = state
        self.array = array


class Bad(Node):
    bitvec: Bitvec

    def __init__(self, nid: int, bitvec: Bitvec, symbol: str = ""):
        super().__init__(nid, symbol)
        self.bitvec = bitvec


class Constraint(Node):
    bitvec: Bitvec

    def __init__(self, nid: int, bitvec: Bitvec, symbol: str = ""):
        super().__init__(nid, symbol)
        self.bitvec = bitvec


class Fair(Node):
    expr: Expr

    def __init__(self, nid: int, expr: Expr, symbol: str = ""):
        super().__init__(nid, symbol)
        self.expr = expr


class Output(Node):
    expr: Expr

    def __init__(self, nid: int, expr: Expr, symbol: str = ""):
        super().__init__(nid, symbol)
        self.expr = expr


class Justice(Node):
    n: int
    expr_list: List[Expr]

    def __init__(self, nid: int, n: int, expr_list: List[Expr], symbol: str = ""):
        super().__init__(nid, symbol)
        self.n = n
        self.expr_list = expr_list


def build_init_function(module: ir.Module, bitvec_states: List[BitvecState],
                        bitvec_inputs: List[BitvecInput],
                        state_struct_type: ir.BaseStructType,
                        input_struct_type: ir.BaseStructType,
                        argument_types: Iterable[ir.Type],
                        generator_maker: Callable[[ir.IRBuilder], InputGenerator]
                        ) -> Tuple[ir.Function, int]:
    function: ir.Function = ir.Function(module, ir.FunctionType(ir.VoidType(), (
        state_struct_type.as_pointer(), input_struct_type.as_pointer(), *argument_types)), 'init')
    builder: ir.IRBuilder = ir.IRBuilder(function.append_basic_block('entry'))
    m: Dict[int, ir.Value] = {}
    generator: InputGenerator = generator_maker(builder)

    i: int
    v: ir.Value
    bitvec_input: BitvecInput
    for i, bitvec_input in enumerate(bitvec_inputs):
        v = generator.generate_input(bitvec_input.sort.width)
        builder.store(v, builder.gep(function.args[1], (ir.Constant(ir.IntType(32), 0),
                                                        ir.Constant(ir.IntType(32), i))))
        m[bitvec_input.nid] = v

    bitvec_state: BitvecState
    for i, bitvec_state in enumerate(bitvec_states):
        if not bitvec_state.init:
            v = generator.generate_input(bitvec_state.sort.width)
        else:
            v = bitvec_state.init.to_ir_value(builder, m)

        builder.store(v, builder.gep(function.args[0], (ir.Constant(ir.IntType(32), 0),
                                                        ir.Constant(ir.IntType(32), i))))
        m[bitvec_state.nid] = v

    builder.ret_void()
    return function, generator.total_used


def build_next_function(module: ir.Module, bitvec_states: Iterable[BitvecState],
                        bitvec_inputs: Iterable[BitvecInput],
                        state_struct_type: ir.BaseStructType,
                        input_struct_type: ir.BaseStructType,
                        argument_types: Iterable[ir.Type],
                        generator_maker: Callable[[ir.IRBuilder], InputGenerator]
                        ) -> Tuple[ir.Function, int]:
    function: ir.Function = ir.Function(module, ir.FunctionType(ir.VoidType(), (
        state_struct_type.as_pointer(), input_struct_type.as_pointer(), *argument_types)), 'next')
    builder: ir.IRBuilder = ir.IRBuilder(function.append_basic_block('entry'))
    ps: List[ir.Value] = [builder.gep(function.args[0], (ir.Constant(ir.IntType(32), 0),
                                                         ir.Constant(ir.IntType(32), i)))
                          for i, _ in enumerate(bitvec_states)]
    m: Dict[int, ir.Value] = dict(zip((bitvec_state.nid for bitvec_state in bitvec_states),
                                      (builder.load(p) for p in ps)))
    generator: InputGenerator = generator_maker(builder)

    i: int
    v: ir.Value
    bitvec_input: BitvecInput
    for i, bitvec_input in enumerate(bitvec_inputs):
        v = generator.generate_input(bitvec_input.sort.width)
        builder.store(v, builder.gep(function.args[1], (ir.Constant(ir.IntType(32), 0),
                                                        ir.Constant(ir.IntType(32), i))))
        m[bitvec_input.nid] = v

    bitvec_State: BitvecState
    p: ir.Value
    for bitvec_state, p in zip(bitvec_states, ps):
        if not bitvec_state.next:
            builder.store(generator.generate_input(bitvec_state.sort.width), p)
        else:
            builder.store(bitvec_state.next.to_ir_value(builder, m), p)

    builder.ret_void()
    return function, generator.total_used


def build_reduce_function(module: ir.Module, name: str, bitvec_states: Iterable[BitvecState],
                          bitvec_inputs: Iterable[BitvecInput],
                          state_struct_type: ir.BaseStructType,
                          input_struct_type: ir.BaseStructType,
                          reduce_function: Callable[[ir.IRBuilder, ir.Value, ir.Value], ir.Value],
                          bool_bitvecs: List[Bitvec], default: bool) -> ir.Function:
    function: ir.Function = ir.Function(module, ir.FunctionType(
        ir.IntType(1), (state_struct_type.as_pointer(), input_struct_type.as_pointer())), name)
    builder: ir.IRBuilder = ir.IRBuilder(function.append_basic_block('entry'))

    if bool_bitvecs:
        m: Dict[int, ir.Value] = dict(itertools.chain((
            (bitvec_state.nid, builder.load(builder.gep(function.args[0], (
                ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), i)))))
            for i, bitvec_state in enumerate(bitvec_states)), (
            (bitvec_input.nid, builder.load(builder.gep(function.args[1], (
                ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), i)))))
            for i, bitvec_input in enumerate(bitvec_inputs))))

        builder.ret(functools.reduce(lambda v1, v2: reduce_function(builder, v1, v2), (
            bitvec.to_ir_value(builder, m) for bitvec in bool_bitvecs)))
        return function
    else:
        builder.ret(ir.Constant(ir.IntType(1), default))
        return function


def build_main_function(module: ir.Module, state_struct_type: ir.BaseStructType,
                        input_struct_type: ir.BaseStructType,
                        init_function: ir.Function, next_function: ir.Function,
                        constraint_function: ir.Function, bad_function: ir.Function,
                        init_tu: int, next_tu: int, n: int) -> ir.Function:
    function: ir.Function = ir.Function(module, ir.FunctionType(ir.IntType(32), ()), 'main')

    entry_block: ir.Block = function.append_basic_block('entry')
    for_body_block: ir.Block = function.append_basic_block('for.body')
    ct1_block: ir.Block = function.append_basic_block('constraint.true')
    bf_block: ir.Block = function.append_basic_block('bad.false')
    for_end_block: ir.Block = function.append_basic_block('for.end')
    ct2_block: ir.Block = function.append_basic_block('constraint.true')
    ret0_block: ir.Block = function.append_basic_block('return.zero')
    ret1_block: ir.Block = function.append_basic_block('return.one')

    entry_builder: ir.IRBuilder = ir.IRBuilder(entry_block)
    entry_builder.call(
        ir.Function(module, ir.FunctionType(ir.VoidType(), (ir.IntType(32),)), 'srand'),
        (entry_builder.trunc(entry_builder.call(
            ir.Function(module, ir.FunctionType(ir.IntType(64), (ir.IntType(64).as_pointer(),)),
                        'time'),
            (ir.Constant(ir.IntType(64).as_pointer(), None),)), ir.IntType(32)),))
    state_ptr: ir.AllocaInstr = entry_builder.alloca(state_struct_type)
    input_ptr: ir.AllocaInstr = entry_builder.alloca(input_struct_type)
    entry_builder.call(init_function, (state_ptr, input_ptr))
    entry_builder.branch(for_body_block)

    for_body_builder: ir.IRBuilder = ir.IRBuilder(for_body_block)
    i_phi: ir.PhiInstr = for_body_builder.phi(ir.IntType(32))
    i_phi.add_incoming(ir_const_int(0, 32), entry_block)
    for_body_builder.cbranch(for_body_builder.call(constraint_function, (state_ptr, input_ptr)),
                             ct1_block, ret0_block)

    ct1_builder: ir.IRBuilder = ir.IRBuilder(ct1_block)
    ct1_builder.cbranch(ct1_builder.call(bad_function, (state_ptr, input_ptr)),
                        ret1_block, bf_block)

    bf_builder: ir.IRBuilder = ir.IRBuilder(bf_block)
    bf_builder.call(next_function, (state_ptr,))
    new_i: ir.Value = bf_builder.add(i_phi, ir_const_int(1, 32))
    i_phi.add_incoming(new_i, bf_block)
    bf_builder.cbranch(bf_builder.icmp_unsigned('<', new_i, ir_const_int(n, 32)), for_body_block,
                       for_end_block)

    for_end_builder: ir.IRBuilder = ir.IRBuilder(for_end_block)
    for_end_builder.cbranch(for_end_builder.call(constraint_function, (state_ptr, input_ptr)),
                            ct2_block, ret0_block)

    ct2_builder: ir.IRBuilder = ir.IRBuilder(ct2_block)
    ct2_builder.cbranch(ct2_builder.call(bad_function, (state_ptr, input_ptr)),
                        ret1_block, ret0_block)

    ret0_builder: ir.IRBuilder = ir.IRBuilder(ret0_block)
    ret0_builder.ret(ir_const_int(0, 32))

    ret1_builder: ir.IRBuilder = ir.IRBuilder(ret1_block)
    ret1_builder.ret(ir_const_int(1, 32))

    return function


def build_test_function(module: ir.Module, state_struct_type: ir.BaseStructType,
                        input_struct_type: ir.BaseStructType,
                        init_function: ir.Function, next_function: ir.Function,
                        constraint_function: ir.Function, bad_function: ir.Function,
                        init_tu: int, next_tu: int, n: int) -> ir.Function:
    function: ir.Function = ir.Function(module, ir.FunctionType(
        ir.IntType(32), (ir.IntType(8).as_pointer(), ir.IntType(64))), 'LLVMFuzzerTestOneInput')
    data: ir.Argument = function.args[0]
    size: ir.Argument = function.args[1]

    ########################################
    # printf_function_type = ir.FunctionType(ir.IntType(32), [
    #     ir.PointerType(ir.IntType(8))], var_arg=True)
    # printf_function = ir.Function(module, printf_function_type, name="printf")
    #
    # fmt = "%x\n\0"
    # c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)), bytearray(fmt.encode("utf8")))
    # global_fmt = ir.GlobalVariable(module, c_fmt.type, name="fstr")
    # global_fmt.linkage = 'internal'
    # global_fmt.global_constant = True
    # global_fmt.initializer = c_fmt
    #
    # fmt_begin = "begin\n\0"
    # c_fmt_begin = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt_begin)),
    #                           bytearray(fmt_begin.encode("utf8")))
    # global_fmt_begin = ir.GlobalVariable(module, c_fmt_begin.type, name="begin_str")
    # global_fmt_begin.linkage = 'internal'
    # global_fmt_begin.global_constant = True
    # global_fmt_begin.initializer = c_fmt_begin
    ########################################

    entry_block: ir.Block = function.append_basic_block('entry')
    init_block: ir.Block = function.append_basic_block('init')
    for_body_block: ir.Block = function.append_basic_block('for.body')
    ct_block: ir.Block = function.append_basic_block('constraint.true')
    bf_block: ir.Block = function.append_basic_block('bad.false')
    next_block: ir.Block = function.append_basic_block('next')
    ret_block: ir.Block = function.append_basic_block('return.zero')
    err_block: ir.Block = function.append_basic_block('error')

    entry_builder: ir.IRBuilder = ir.IRBuilder(entry_block)
    ########################################
    # fmt_arg = entry_builder.bitcast(global_fmt, ir.IntType(8).as_pointer())
    # fmt_begin_arg = entry_builder.bitcast(global_fmt_begin, ir.IntType(8).as_pointer())
    # entry_builder.call(printf_function, (fmt_begin_arg, size))
    # entry_builder.call(printf_function, (fmt_arg, size))
    # entry_builder.call(printf_function, (fmt_arg, entry_builder.load(
    #     entry_builder.gep(data, (ir.Constant(ir.IntType(32), 0),)))))
    # entry_builder.call(printf_function, (fmt_arg, entry_builder.load(
    #     entry_builder.gep(data, (ir.Constant(ir.IntType(32), 1),)))))
    ########################################
    entry_builder.cbranch(
        entry_builder.icmp_unsigned('<', size, ir.Constant(ir.IntType(64), init_tu)),
        ret_block, init_block)

    init_builder: ir.IRBuilder = ir.IRBuilder(init_block)
    state_ptr: ir.AllocaInstr = init_builder.alloca(state_struct_type)
    input_ptr: ir.AllocaInstr = init_builder.alloca(input_struct_type)
    init_builder.call(init_function, (state_ptr, input_ptr, data, ir_const_int(0, 64)))
    ########################################
    # init_builder.call(printf_function, (
    #     fmt_arg, init_builder.load(gep(init_builder, state_ptr, 0)),))
    # init_builder.call(printf_function, (
    #     fmt_arg, init_builder.load(gep(init_builder, state_ptr, 1)),))
    ########################################
    init_builder.branch(for_body_block)

    for_body_builder: ir.IRBuilder = ir.IRBuilder(for_body_block)
    u_phi: ir.PhiInstr = for_body_builder.phi(ir.IntType(64))
    u_phi.add_incoming(ir_const_int(init_tu, 64), init_block)

    for_body_builder.cbranch(for_body_builder.call(
        constraint_function, (state_ptr, input_ptr)), ct_block, ret_block)

    ct1_builder: ir.IRBuilder = ir.IRBuilder(ct_block)
    ########################################
    # bad_builder.call(printf_function, (
    #     fmt_arg, bad_builder.load(gep(bad_builder, state_ptr, 0)),))
    # bad_builder.call(printf_function, (
    #     fmt_arg, bad_builder.load(gep(bad_builder, state_ptr, 1)),))
    ########################################
    ct1_builder.cbranch(ct1_builder.call(bad_function, (state_ptr, input_ptr)), err_block, bf_block)

    bf_builder: ir.IRBuilder = ir.IRBuilder(bf_block)
    new_u: ir.Value = bf_builder.add(u_phi, ir_const_int(next_tu, 64))
    bf_builder.cbranch(bf_builder.icmp_unsigned('<', size, new_u), ret_block, next_block)

    next_builder: ir.IRBuilder = ir.IRBuilder(next_block)
    ########################################
    # next_builder.call(printf_function, (
    #     fmt_arg, next_builder.load(gep(next_builder, state_ptr, 0)),))
    # next_builder.call(printf_function, (
    #     fmt_arg, next_builder.load(gep(next_builder, state_ptr, 1)),))
    ########################################
    next_builder.call(next_function, (state_ptr, input_ptr, data, u_phi))
    ########################################
    # next_builder.call(printf_function, (
    #     fmt_arg, next_builder.load(gep(next_builder, state_ptr, 0)),))
    # next_builder.call(printf_function, (
    #     fmt_arg, next_builder.load(gep(next_builder, state_ptr, 1)),))
    ########################################
    u_phi.add_incoming(new_u, next_block)
    next_builder.branch(for_body_block)

    ret_builder: ir.IRBuilder = ir.IRBuilder(ret_block)
    ret_builder.ret(ir_const_int(0, 32))

    err_builder: ir.IRBuilder = ir.IRBuilder(err_block)
    err_builder.call(ir.Function(module, ir.FunctionType(ir.VoidType(), (ir.IntType(32),)), 'exit'),
                     (ir_const_int(1, 32),))
    err_builder.unreachable()

    return function


class Btor2Parser:
    bitvec_sort_table: Dict[int, BitvecSort]
    array_sort_table: Dict[int, ArraySort]
    bitvec_state_table: Dict[int, BitvecState]
    array_state_table: Dict[int, ArrayState]
    bitvec_input_table: Dict[int, BitvecInput]
    array_input_table: Dict[int, ArrayInput]
    bitvec_table: Dict[int, Bitvec]
    array_table: Dict[int, Array]
    bad_list: List[Bad]
    constraint_list: List[Constraint]
    fair_list: List[Fair]
    output_list: List[Output]
    justice_list: List[Justice]

    def __init__(self):
        self.bitvec_sort_table = {}
        self.bitvec_state_table = {}
        self.bitvec_input_table = {}
        self.bitvec_table = {}
        self.array_sort_table = {}
        self.array_input_table = {}
        self.array_state_table = {}
        self.array_table = {}
        self.bad_list = []
        self.constraint_list = []
        self.fair_list = []
        self.output_list = []
        self.justice_list = []

    def get_sort(self, s: Union[int, str]) -> Sort:
        sid: int = int(s)
        return self.bitvec_sort_table[sid] if sid in self.bitvec_sort_table else \
            self.array_sort_table[sid]

    def get_bitvec_sort(self, s: Union[int, str]) -> BitvecSort:
        return self.bitvec_sort_table[int(s)]

    def get_array_sort(self, s: Union[int, str]) -> ArraySort:
        return self.array_sort_table[int(s)]

    def get_expr(self, n: Union[int, str]) -> Expr:
        nid: int = int(n)
        if nid < 0:
            return Minus(self.bitvec_table[-nid])
        return self.bitvec_table[nid] if nid in self.bitvec_table else self.array_table[nid]

    def get_bitvec_state(self, n: Union[int, str]) -> BitvecState:
        return self.bitvec_state_table[int(n)]

    def get_array_state(self, n: Union[int, str]) -> ArrayState:
        return self.array_state_table[int(n)]

    def get_bitvec(self, n: Union[int, str]) -> Bitvec:
        nid: int = int(n)
        if nid < 0:
            return Minus(self.bitvec_table[-nid])
        return self.bitvec_table[nid]

    def get_array(self, n: Union[int, str]) -> Array:
        return self.array_table[int(n)]

    def parse(self, source: TextIO) -> None:
        for line in source:
            line_left: str
            _: str
            line_left, _, _ = line.partition(';')
            tokens: List[str] = line_left.split()

            if len(tokens) == 0:
                continue

            name: str = tokens[1]
            if name == 'sort':
                sid: int = int(tokens[0])
                if tokens[2] == 'array':
                    self.array_sort_table[sid] = ArraySort(sid, self.get_sort(tokens[3]),
                                                           self.get_sort(tokens[4]))
                elif tokens[2] == 'bitvec':
                    self.bitvec_sort_table[sid] = BitvecSort(sid, int(tokens[3]))
                continue

            nid: int = int(tokens[0])

            if name == 'bad':
                self.bad_list.append(Bad(nid, self.get_bitvec(tokens[2])))
                continue
            if name == 'constraint':
                self.constraint_list.append(Constraint(nid, self.get_bitvec(tokens[2])))
                continue
            if name == 'fair':
                self.fair_list.append(Fair(nid, self.get_expr(tokens[2])))
                continue
            if name == 'output':
                self.output_list.append(Output(nid, self.get_expr(tokens[2])))
                continue
            if name == 'justice':
                n: int = int(tokens[2])
                self.justice_list.append(
                    Justice(nid, n, [self.get_expr(x) for x in tokens[3:3 + n]]))
                continue

            # noinspection DuplicatedCode
            if name == 'read':
                read_sid: int = int(tokens[2])
                if read_sid in self.bitvec_sort_table:
                    self.bitvec_table[nid] = BitvecRead(nid, self.get_bitvec_sort(read_sid),
                                                        self.get_array(tokens[3]),
                                                        self.get_expr(tokens[4]))
                elif read_sid in self.array_sort_table:
                    self.array_table[nid] = ArrayRead(nid, self.get_array_sort(read_sid),
                                                      self.get_array(tokens[3]),
                                                      self.get_expr(tokens[4]))
                continue
            if name == 'state':
                state_sid: int = int(tokens[2])
                if state_sid in self.bitvec_sort_table:
                    bitvec_state: BitvecState = BitvecState(nid, self.get_bitvec_sort(state_sid))
                    self.bitvec_state_table[nid] = self.bitvec_table[nid] = bitvec_state
                elif state_sid in self.array_sort_table:
                    array_state: ArrayState = ArrayState(nid, self.get_array_sort(state_sid))
                    self.array_state_table[nid] = self.array_table[nid] = array_state
                continue
            if name == 'input':
                input_sid: int = int(tokens[2])
                if input_sid in self.bitvec_sort_table:
                    bitvec_input: BitvecInput = BitvecInput(nid, self.get_bitvec_sort(input_sid))
                    self.bitvec_input_table[nid] = self.bitvec_table[nid] = bitvec_input
                elif input_sid in self.array_sort_table:
                    array_input: ArrayInput = ArrayInput(nid, self.get_array_sort(input_sid))
                    self.array_input_table[nid] = self.array_table[nid] = array_input
                continue
            if name == 'init':
                init_sid: int = int(tokens[2])
                if init_sid in self.bitvec_sort_table:
                    self.get_bitvec_state(tokens[3]).init = self.get_bitvec(tokens[4])
                elif init_sid in self.array_sort_table:
                    self.get_array_state(tokens[3]).init = self.get_expr(tokens[4])
                continue
            if name == 'next':
                next_sid: int = int(tokens[2])
                if next_sid in self.bitvec_sort_table:
                    self.get_bitvec_state(tokens[3]).next = self.get_bitvec(tokens[4])
                elif next_sid in self.array_sort_table:
                    self.get_array_state(tokens[3]).next = self.get_array(tokens[4])
                continue
            if name == 'write':
                self.array_table[nid] = Write(nid, self.get_array_sort(int(tokens[2])),
                                              self.get_array(tokens[3]),
                                              self.get_expr(tokens[4]), self.get_expr(tokens[5]))
                continue
            if name == "ite":
                ite_sid: int = int(tokens[2])
                if ite_sid in self.bitvec_sort_table:
                    self.bitvec_table[nid] = BitvecIte(nid, self.bitvec_sort_table[ite_sid],
                                                       self.get_bitvec(tokens[3]),
                                                       self.get_bitvec(tokens[4]),
                                                       self.get_bitvec(tokens[5]))
                elif ite_sid in self.array_sort_table:
                    self.array_table[nid] = ArrayIte(nid, self.array_sort_table[ite_sid],
                                                     self.get_bitvec(tokens[3]),
                                                     self.get_array(tokens[4]),
                                                     self.get_array(tokens[5]))
                continue

            sort: BitvecSort = self.get_bitvec_sort(tokens[2])
            if name == 'one':
                self.bitvec_table[nid] = One(nid, sort)
            elif name == 'ones':
                self.bitvec_table[nid] = Ones(nid, sort)
            elif name == 'zero':
                self.bitvec_table[nid] = Zero(nid, sort)
            elif name == 'const':
                self.bitvec_table[nid] = Const(nid, sort, tokens[3])
            elif name == 'constd':
                self.bitvec_table[nid] = Constd(nid, sort, tokens[3])
            elif name == 'consth':
                self.bitvec_table[nid] = Consth(nid, sort, tokens[3])
            elif name == 'sext':
                self.bitvec_table[nid] = Sext(nid, sort, self.get_bitvec(tokens[3]), int(tokens[4]))
            elif name == 'uext':
                self.bitvec_table[nid] = Uext(nid, sort, self.get_bitvec(tokens[3]), int(tokens[4]))
            elif name == 'slice':
                self.bitvec_table[nid] = Slice(nid, sort, self.get_bitvec(tokens[3]),
                                               int(tokens[4]), int(tokens[5]))
            elif name == 'not':
                self.bitvec_table[nid] = Not(nid, sort, self.get_bitvec(tokens[3]))
            elif name == 'inc':
                self.bitvec_table[nid] = Inc(nid, sort, self.get_bitvec(tokens[3]))
            elif name == 'dec':
                self.bitvec_table[nid] = Dec(nid, sort, self.get_bitvec(tokens[3]))
            elif name == 'neg':
                self.bitvec_table[nid] = Neg(nid, sort, self.get_bitvec(tokens[3]))
            elif name == 'redand':
                self.bitvec_table[nid] = Redand(nid, sort, self.get_bitvec(tokens[3]))
            elif name == 'redor':
                self.bitvec_table[nid] = Redor(nid, sort, self.get_bitvec(tokens[3]))
            elif name == 'redxor':
                self.bitvec_table[nid] = Redxor(nid, sort, self.get_bitvec(tokens[3]))
            elif name == 'iff':
                self.bitvec_table[nid] = Iff(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'implies':
                self.bitvec_table[nid] = Implies(nid, sort, self.get_bitvec(tokens[3]),
                                                 self.get_bitvec(tokens[4]))
            elif name == 'eq':
                self.bitvec_table[nid] = Eq(nid, sort, self.get_expr(tokens[3]),
                                            self.get_expr(tokens[4]))
            elif name == 'neq':
                self.bitvec_table[nid] = Neq(nid, sort, self.get_expr(tokens[3]),
                                             self.get_expr(tokens[4]))
            elif name == 'sgt':
                self.bitvec_table[nid] = Sgt(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'ugt':
                self.bitvec_table[nid] = Ugt(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'sgte':
                self.bitvec_table[nid] = Sgte(nid, sort, self.get_bitvec(tokens[3]),
                                              self.get_bitvec(tokens[4]))
            elif name == 'ugte':
                self.bitvec_table[nid] = Ugte(nid, sort, self.get_bitvec(tokens[3]),
                                              self.get_bitvec(tokens[4]))
            elif name == 'slt':
                self.bitvec_table[nid] = Slt(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'ult':
                self.bitvec_table[nid] = Ult(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'slte':
                self.bitvec_table[nid] = Slte(nid, sort, self.get_bitvec(tokens[3]),
                                              self.get_bitvec(tokens[4]))
            elif name == 'ulte':
                self.bitvec_table[nid] = Ulte(nid, sort, self.get_bitvec(tokens[3]),
                                              self.get_bitvec(tokens[4]))
            elif name == 'and':
                self.bitvec_table[nid] = And(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'nand':
                self.bitvec_table[nid] = Nand(nid, sort, self.get_bitvec(tokens[3]),
                                              self.get_bitvec(tokens[4]))
            elif name == 'nor':
                self.bitvec_table[nid] = Nor(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'or':
                self.bitvec_table[nid] = Or(nid, sort, self.get_bitvec(tokens[3]),
                                            self.get_bitvec(tokens[4]))
            elif name == 'xnor':
                self.bitvec_table[nid] = Xnor(nid, sort, self.get_bitvec(tokens[3]),
                                              self.get_bitvec(tokens[4]))
            elif name == 'xor':
                self.bitvec_table[nid] = Xor(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'rol':
                self.bitvec_table[nid] = Rol(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'ror':
                self.bitvec_table[nid] = Ror(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'sll':
                self.bitvec_table[nid] = Sll(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'sra':
                self.bitvec_table[nid] = Sra(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'srl':
                self.bitvec_table[nid] = Srl(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'add':
                self.bitvec_table[nid] = Add(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'mul':
                self.bitvec_table[nid] = Mul(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'sdiv':
                self.bitvec_table[nid] = Sdiv(nid, sort, self.get_bitvec(tokens[3]),
                                              self.get_bitvec(tokens[4]))
            elif name == 'udiv':
                self.bitvec_table[nid] = Udiv(nid, sort, self.get_bitvec(tokens[3]),
                                              self.get_bitvec(tokens[4]))
            elif name == 'smod':
                self.bitvec_table[nid] = Smod(nid, sort, self.get_bitvec(tokens[3]),
                                              self.get_bitvec(tokens[4]))
            elif name == 'srem':
                self.bitvec_table[nid] = Srem(nid, sort, self.get_bitvec(tokens[3]),
                                              self.get_bitvec(tokens[4]))
            elif name == 'urem':
                self.bitvec_table[nid] = Urem(nid, sort, self.get_bitvec(tokens[3]),
                                              self.get_bitvec(tokens[4]))
            elif name == 'sub':
                self.bitvec_table[nid] = Sub(nid, sort, self.get_bitvec(tokens[3]),
                                             self.get_bitvec(tokens[4]))
            elif name == 'saddo':
                self.bitvec_table[nid] = Saddo(nid, sort, self.get_bitvec(tokens[3]),
                                               self.get_bitvec(tokens[4]))
            elif name == 'uaddo':
                self.bitvec_table[nid] = Uaddo(nid, sort, self.get_bitvec(tokens[3]),
                                               self.get_bitvec(tokens[4]))
            elif name == 'sdivo':
                self.bitvec_table[nid] = Sdivo(nid, sort, self.get_bitvec(tokens[3]),
                                               self.get_bitvec(tokens[4]))
            elif name == 'udivo':
                self.bitvec_table[nid] = Udivo(nid, sort, self.get_bitvec(tokens[3]),
                                               self.get_bitvec(tokens[4]))
            elif name == 'smulo':
                self.bitvec_table[nid] = Smulo(nid, sort, self.get_bitvec(tokens[3]),
                                               self.get_bitvec(tokens[4]))
            elif name == 'umulo':
                self.bitvec_table[nid] = Umulo(nid, sort, self.get_bitvec(tokens[3]),
                                               self.get_bitvec(tokens[4]))
            elif name == 'ssubo':
                self.bitvec_table[nid] = Ssubo(nid, sort, self.get_bitvec(tokens[3]),
                                               self.get_bitvec(tokens[4]))
            elif name == 'usubo':
                self.bitvec_table[nid] = Usubo(nid, sort, self.get_bitvec(tokens[3]),
                                               self.get_bitvec(tokens[4]))
            elif name == 'concat':
                self.bitvec_table[nid] = Concat(nid, sort, self.get_bitvec(tokens[3]),
                                                self.get_bitvec(tokens[4]))

    def build_functions(self, module: ir.Module, n: int,
                        argument_types: Iterable[ir.Type],
                        generator_maker: Callable[[ir.IRBuilder], InputGenerator],
                        entry_function_builder: Callable[
                            [ir.Module, ir.BaseStructType, ir.BaseStructType, ir.Function,
                             ir.Function, ir.Function, ir.Function, int, int, int], ir.Function]
                        ) -> None:
        bitvec_states: List[BitvecState] = list(self.bitvec_state_table.values())
        bitvec_inputs: List[BitvecInput] = list(self.bitvec_input_table.values())

        bitvec_states.sort(key=lambda x: x.nid)
        bitvec_inputs.sort(key=lambda x: x.nid)

        state_struct_type: ir.IdentifiedStructType = module.context.get_identified_type(
            'struct.State')
        state_struct_type.set_body(*(e.sort.to_ir_type() for e in bitvec_states))
        input_struct_type: ir.IdentifiedStructType = module.context.get_identified_type(
            'struct.Input')
        input_struct_type.set_body(*(e.sort.to_ir_type() for e in bitvec_inputs))

        init_function, init_tu = build_init_function(
            module, bitvec_states, bitvec_inputs, state_struct_type, input_struct_type,
            argument_types, generator_maker)

        next_function, next_tu = build_next_function(
            module, bitvec_states, bitvec_inputs, state_struct_type, input_struct_type,
            argument_types, generator_maker)

        bad_function = build_reduce_function(
            module, 'bad', bitvec_states, bitvec_inputs, state_struct_type,
            input_struct_type, ir.IRBuilder.or_, [bad.bitvec for bad in self.bad_list], False)

        constraint_function = build_reduce_function(
            module, 'constraint', bitvec_states, bitvec_inputs, state_struct_type,
            input_struct_type, ir.IRBuilder.and_,
            [constraint.bitvec for constraint in self.constraint_list], True)

        entry_function_builder(module, state_struct_type, input_struct_type, init_function,
                               next_function, constraint_function, bad_function, init_tu, next_tu,
                               n)

    def build_module_with_main(self, name: str, n: int, triple: str, data_layout: str) -> ir.Module:
        module: ir.Module = ir.Module(name, ir.Context())
        module.triple = triple
        module.data_layout = data_layout
        rand_function: ir.Function = ir.Function(module, ir.FunctionType(ir.IntType(32), ()),
                                                 'rand')

        def make_generator(builder: ir.IRBuilder) -> InputGenerator:
            return RandInputGenerator(builder, rand_function)

        self.build_functions(module, n, (), make_generator, build_main_function)
        return module

    def build_module_with_test(self, name: str, triple: str, data_layout: str) -> ir.Module:
        module: ir.Module = ir.Module(name, ir.Context())
        module.triple = triple
        module.data_layout = data_layout

        def make_generator(builder: ir.IRBuilder) -> InputGenerator:
            return FuzzerInputGenerator(builder, builder.function.args[2], builder.function.args[3])

        self.build_functions(module, 0, (ir.IntType(8).as_pointer(), ir.IntType(64)),
                             make_generator, build_test_function)
        return module


default_data_layout: str = ""


def get_default_data_layout() -> str:
    global default_data_layout
    if not default_data_layout:
        binding.initialize_native_target()
        target: binding.Target = binding.Target.from_default_triple()
        default_data_layout = target.create_target_machine().target_data
    return default_data_layout


def btor2llvm_main(input_path: str, output_path: str, n: int,
                   triple: str = binding.get_default_triple(),
                   data_layout: str = get_default_data_layout()) -> None:
    parser: Btor2Parser = Btor2Parser()
    with open(input_path) as input_file:
        parser.parse(input_file)

    module: ir.Module = parser.build_module_with_main(input_path, n, triple, data_layout)
    with open(output_path, 'w+') as output_file:
        output_file.write(str(module))


def btor2llvm_test(input_path: str, output_path: str, triple: str = binding.get_default_triple(),
                   data_layout: str = get_default_data_layout()) -> None:
    parser: Btor2Parser = Btor2Parser()
    with open(input_path) as input_file:
        parser.parse(input_file)

    module: ir.Module = parser.build_module_with_test(input_path, triple, data_layout)
    with open(output_path, 'w+') as output_file:
        output_file.write(str(module))


def main() -> int:
    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='A tool to convert btor2 files to LLVM.')
    argument_parser.add_argument('input', metavar='IN_FILE',
                                 help='Specify the path to the input file.')
    argument_parser.add_argument('-output', '--output', metavar='OUT_FILE',
                                 help='Set the path to the output file. (default: out.ll)',
                                 default='out.ll')
    argument_parser.add_argument('-n', '--n',
                                 help='Set the number of iterations in the main function. '
                                      '(default: 10)', type=int, default=10)
    argument_parser.add_argument('-mode', '--mode',
                                 help='If set to "test", generate the '
                                      '`LLVMFuzzerTestOneInput` function. '
                                      'If set to "main", generate the `main` function. '
                                      '(default: test)', choices=("test", "main"),
                                 default="test")
    argument_parser.add_argument('-datalayout', '--datalayout', type=str,
                                 help='Set the datalayout.')
    argument_parser.add_argument('-triple', '--triple', type=str,
                                 help='Set the triple.')

    namespace: argparse.Namespace = argument_parser.parse_args()

    if namespace.mode == 'main':
        btor2llvm_main(namespace.input, namespace.output, namespace.n)
    else:
        btor2llvm_test(namespace.input, namespace.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
