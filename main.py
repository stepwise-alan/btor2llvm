import argparse
import itertools
import sys
from abc import ABC
from typing import Optional, List, TextIO, Union, Dict, ValuesView, Iterable

from llvmlite import ir  # type: ignore


class Node(ABC):
    symbol: str

    def __init__(self, symbol: str = ""):
        self.symbol = symbol


class Sort(Node):
    sid: int

    def __init__(self, sid: int, symbol: str = ""):
        super().__init__(symbol)
        self.sid = sid

    def to_ir_type(self) -> ir.Type:
        raise NotImplementedError


class BitvecSort(Sort):
    width: int

    def __init__(self, sid: int, width: int, symbol: str = ""):
        super().__init__(sid, symbol)
        self.width = width

    def to_ir_type(self) -> ir.Type:
        return ir.IntType(self.width)


class ArraySort(Sort):
    index_sort: Sort
    element_sort: Sort

    def __init__(self, sid: int, index_sort: Sort, element_sort: Sort, symbol: str = ""):
        super().__init__(sid, symbol)
        self.index_sort = index_sort
        self.element_sort = element_sort

    def to_ir_type(self) -> ir.ArrayType:
        # if isinstance(self.index_sort, BitvecSort):
        #     return ir.ArrayType(self.element_sort.to_ir_type(), 1 << self.index_sort.width)
        # TODO: support array as index
        raise NotImplementedError


class Expr(Node):
    nid: int
    sort: Sort

    def __init__(self, nid: int, sort: Sort, symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.sort = sort

    def to_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        if self.nid in m:
            return m[self.nid]
        return self.to_new_ir_value(builder, m)

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise NotImplementedError


class Bitvec(Expr, ABC):
    sort: BitvecSort

    def __init__(self, nid: int, sort: BitvecSort, symbol: str = ""):
        super().__init__(nid, sort, symbol)


class Array(Expr, ABC):
    sort: ArraySort

    def __init__(self, nid: int, sort: ArraySort, symbol: str = ""):
        super().__init__(nid, sort, symbol)


class Minus(Bitvec):
    bitvec: Bitvec

    def __init__(self, bitvec: Bitvec):
        super().__init__(-bitvec.nid, bitvec.sort)
        self.bitvec = bitvec

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        return builder.not_(self.bitvec.to_ir_value(builder, m))


class BitvecInput(Bitvec):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise ValueError


class ArrayInput(Array):
    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise ValueError


class Constant(Bitvec):
    value: int

    def __init__(self, nid: int, sort: BitvecSort, value: int, symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.value = value

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


class State:
    init: Optional[Expr]
    next: Optional[Expr]

    def __init__(self):
        self.init = None
        self.next = None


class BitvecState(Bitvec, State):
    init: Optional[Bitvec]
    next: Optional[Bitvec]

    def __init__(self, nid: int, sort: BitvecSort, symbol: str = ""):
        Bitvec.__init__(self, nid, sort, symbol)
        State.__init__(self)

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise ValueError


class ArrayState(Array, State):
    next: Optional[Array]

    def __init__(self, nid: int, sort: ArraySort, symbol: str = ""):
        Array.__init__(self, nid, sort, symbol)
        State.__init__(self)

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise ValueError


class Ext(Bitvec, ABC):
    bitvec: Bitvec
    w: int

    def __init__(self, nid: int, sort: BitvecSort, bitvec: Bitvec, w: int, symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.bitvec = bitvec
        self.w = w


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
        return builder.and_(builder.icmp_signed('==', self.bitvec1.to_ir_value(builder, m),
                                                1 << self.bitvec1.sort.width - 1),
                            builder.icmp_signed('==', self.bitvec2.to_ir_value(builder, m), -1))


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
        return builder.or_(
            builder.shl(
                builder.zext(self.bitvec1.to_ir_value(builder, m), self.sort.to_ir_type()),
                ir.Constant(self.bitvec1.sort.to_ir_type(), self.bitvec2.sort.width)),
            builder.zext(self.bitvec2.to_ir_value(builder, m), self.sort.to_ir_type()))


class Read:
    array: Array
    index_expr: Expr

    def __init__(self, array: Array, index_expr: Expr):
        self.array = array
        self.index_expr = index_expr


class BitvecRead(Bitvec, Read):
    def __init__(self, nid: int, sort: BitvecSort, array: Array, index_expr: Expr,
                 symbol: str = ""):
        Bitvec.__init__(self, nid, sort, symbol)
        Read.__init__(self, array, index_expr)

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise NotImplementedError


class ArrayRead(Array, Read):
    def __init__(self, nid: int, sort: ArraySort, array: Array, index_expr: Expr, symbol: str = ""):
        Array.__init__(self, nid, sort, symbol)
        Read.__init__(self, array, index_expr)

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise NotImplementedError


class Ite:
    cond_bitvec: Bitvec
    then_expr: Expr
    else_expr: Expr

    def __init__(self, cond_bitvec: Bitvec, then_expr: Expr, else_expr: Expr):
        self.cond_bitvec = cond_bitvec
        self.then_expr = then_expr
        self.else_expr = else_expr


class ArrayIte(Array, Ite):
    then_expr: Array
    else_expr: Array

    def __init__(self, nid: int, sort: ArraySort, cond_bitvec: Bitvec, then_expr: Array,
                 else_expr: Array, symbol: str = ""):
        Array.__init__(self, nid, sort, symbol)
        Ite.__init__(self, cond_bitvec, then_expr, else_expr)

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise NotImplementedError


class BitvecIte(Bitvec, Ite):
    then_expr: Bitvec
    else_expr: Bitvec

    def __init__(self, nid: int, sort: BitvecSort, cond_bitvec: Bitvec, then_expr: Bitvec,
                 else_expr: Bitvec, symbol: str = ""):
        Bitvec.__init__(self, nid, sort, symbol)
        Ite.__init__(self, cond_bitvec, then_expr, else_expr)

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

    def to_new_ir_value(self, builder: ir.IRBuilder, m: Dict[int, ir.Value]) -> ir.Value:
        raise NotImplementedError


class BitvecInit(Node):
    nid: int
    sort: BitvecSort
    state: BitvecState
    bitvec: Bitvec

    def __init__(self, nid: int, sort: BitvecSort, state: BitvecState, bitvec: Bitvec,
                 symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.sort = sort
        self.state = state
        self.bitvec = bitvec


class ArrayInit(Node):
    nid: int
    sort: ArraySort
    state: ArrayState
    array: Array

    def __init__(self, nid: int, sort: ArraySort, state: ArrayState, bitvec: Bitvec,
                 symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.sort = sort
        self.state = state
        self.bitvec = bitvec


class BitvecNext(Node):
    nid: int
    sort: BitvecSort
    state: BitvecState
    bitvec: Bitvec

    def __init__(self, nid: int, sort: BitvecSort, state: BitvecState, bitvec: Bitvec,
                 symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.sort = sort
        self.state = state
        self.bitvec = bitvec


class ArrayNext(Node):
    nid: int
    sort: ArraySort
    state: ArrayState
    array: Array

    def __init__(self, nid: int, sort: ArraySort, state: ArrayState, bitvec: Bitvec,
                 symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.sort = sort
        self.state = state
        self.bitvec = bitvec


class Bad(Node):
    nid: int
    bitvec: Bitvec

    def __init__(self, nid: int, bitvec: Bitvec, symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.bitvec = bitvec


class Constraint(Node):
    nid: int
    bitvec: Bitvec

    def __init__(self, nid: int, bitvec: Bitvec, symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.bitvec = bitvec


class Fair(Node):
    nid: int
    expr: Expr

    def __init__(self, nid: int, expr: Expr, symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.expr = expr


class Output(Node):
    nid: int
    expr: Expr

    def __init__(self, nid: int, expr: Expr, symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.expr = expr


class Justice(Node):
    nid: int
    n: int
    expr_list: List[Expr]

    def __init__(self, nid: int, n: int, expr_list: List[Expr], symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.n = n
        self.expr_list = expr_list


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

    def get_function_arg_types(self) -> Iterable[ir.Type]:
        return (v.sort.to_ir_type() for v in itertools.chain(
            self.bitvec_state_table.values(), self.bitvec_input_table.values()))

    def get_function_type(self) -> ir.FunctionType:
        states: ValuesView[BitvecState] = self.bitvec_state_table.values()
        state_ir_types: List[ir.Type] = [v.sort.to_ir_type() for v in states]
        ret_type: ir.LiteralStructType = ir.LiteralStructType(state_ir_types)
        return ir.FunctionType(ret_type, self.get_function_arg_types())

    def build_init_function(self, module: ir.Module) -> ir.Function:
        function_type: ir.FunctionType = self.get_function_type()
        function: ir.Function = ir.Function(module, function_type, 'init')
        builder: ir.IRBuilder = ir.IRBuilder(function.append_basic_block('entry'))
        m: Dict[int, ir.Value] = dict(zip(itertools.chain(self.bitvec_state_table.keys(),
                                                          self.bitvec_input_table.keys()),
                                          function.args))
        s: ir.Value = ir.Constant(function_type.return_type, ir.Undefined)
        for i, state in enumerate(self.bitvec_state_table.values()):
            s = builder.insert_value(s,
                                     state.init.to_ir_value(builder, m) if state.init else
                                     ir.Constant(state.sort.to_ir_type(), ir.Undefined), i)
        builder.ret(s)
        return function

    def build_next_function(self, module: ir.Module) -> ir.Function:
        function_type: ir.FunctionType = self.get_function_type()
        function: ir.Function = ir.Function(module, function_type, 'next')
        builder: ir.IRBuilder = ir.IRBuilder(function.append_basic_block('entry'))
        m: Dict[int, ir.Value] = dict(zip(itertools.chain(self.bitvec_state_table.keys(),
                                                          self.bitvec_input_table.keys()),
                                          function.args))
        s: ir.Value = ir.Constant(function_type.return_type, ir.Undefined)
        for i, state in enumerate(self.bitvec_state_table.values()):
            s = builder.insert_value(s,
                                     state.next.to_ir_value(builder, m) if state.next else
                                     ir.Constant(state.sort.to_ir_type(), ir.Undefined), i)
        builder.ret(s)
        return function

    def build_bad_function(self, module: ir.Module) -> ir.Function:
        function: ir.Function = ir.Function(
            module, ir.FunctionType(ir.IntType(1), self.get_function_arg_types()), 'bad')
        builder: ir.IRBuilder = ir.IRBuilder(function.append_basic_block('entry'))
        m: Dict[int, ir.Value] = dict(zip(itertools.chain(self.bitvec_state_table.keys(),
                                                          self.bitvec_input_table.keys()),
                                          function.args))
        s: ir.Value = ir.Constant(ir.IntType(1), False)
        for bad in self.bad_list:
            s = builder.or_(s, bad.bitvec.to_ir_value(builder, m))
        builder.ret(s)
        return function

    def build_constraint_function(self, module: ir.Module) -> ir.Function:
        function: ir.Function = ir.Function(
            module, ir.FunctionType(ir.IntType(1), self.get_function_arg_types()), 'constraint')
        builder: ir.IRBuilder = ir.IRBuilder(function.append_basic_block('entry'))
        m: Dict[int, ir.Value] = dict(zip(itertools.chain(self.bitvec_state_table.keys(),
                                                          self.bitvec_input_table.keys()),
                                          function.args))
        s: ir.Value = ir.Constant(ir.IntType(1), True)
        for constraint in self.constraint_list:
            s = builder.and_(s, constraint.bitvec.to_ir_value(builder, m))
        builder.ret(s)
        return function

    def build_module(self) -> ir.Module:
        module: ir.Module = ir.Module(__file__)
        self.build_init_function(module)
        self.build_next_function(module)
        self.build_bad_function(module)
        self.build_constraint_function(module)
        return module


def main() -> None:
    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='A tool to convert btor2 files to LLVM.')
    argument_parser.add_argument('input', metavar='FILE', help='input btor2 file', nargs='?',
                                 type=argparse.FileType('r'), default=sys.stdin)
    argument_parser.add_argument('-output', '--output', metavar='FILE',
                                 help='place the output into [FILE]',
                                 nargs='?', type=argparse.FileType('w+'), default=sys.stdout)

    namespace: argparse.Namespace = argument_parser.parse_args()

    parser: Btor2Parser = Btor2Parser()
    parser.parse(namespace.input)
    namespace.output.write(str(parser.build_module()))

    if namespace.input != sys.stdin:
        namespace.input.close()
    if namespace.output != sys.stdout:
        namespace.output.close()


if __name__ == '__main__':
    main()
