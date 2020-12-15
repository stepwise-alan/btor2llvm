from dataclasses import dataclass, InitVar, field
from typing import Optional, List


@dataclass(frozen=True)
class Node:
    nid: int
    symbol: str
    comment: str


@dataclass(frozen=True)
class Sort(Node):
    @property
    def sid(self) -> int:
        return self.nid


@dataclass(frozen=True)
class BitvecSort(Sort):
    width: int


@dataclass(frozen=True)
class ArraySort(Sort):
    index_sort: Sort
    element_sort: Sort


@dataclass(frozen=True)
class Expr(Node):
    sort: Sort


@dataclass(frozen=True)
class Bitvec(Expr):
    sort: BitvecSort


@dataclass(frozen=True)
class Array(Expr):
    sort: ArraySort


@dataclass(frozen=True)
class Minus(Bitvec):
    bitvec: Bitvec

    def __init__(self, bitvec: Bitvec):
        super().__init__(-bitvec.nid, "", "", bitvec.sort)
        self.__dict__['bitvec'] = bitvec


@dataclass(frozen=True)
class Input(Expr):
    pass


@dataclass(frozen=True)
class BitvecInput(Bitvec, Input):
    pass


@dataclass(frozen=True)
class ArrayInput(Array, Input):
    pass


@dataclass(frozen=True)
class Constant(Bitvec):
    value: int


@dataclass(frozen=True)
class One(Constant):
    value: int = 1


@dataclass(frozen=True)
class Ones(Constant):
    value: int = -1


@dataclass(frozen=True)
class Zero(Constant):
    value: int = 0


@dataclass(frozen=True)
class Const(Constant):
    value: int = field(init=False)
    bin_str: InitVar[str] = None

    def __post_init__(self, bin_str: str):
        self.__dict__['value'] = int(bin_str, 2)


@dataclass(frozen=True)
class Constd(Constant):
    value: int = field(init=False)
    dec_str: InitVar[str] = None

    def __post_init__(self, dec_str: str):
        self.__dict__['value'] = int(dec_str, 10)


@dataclass(frozen=True)
class Consth(Constant):
    value: int = field(init=False)
    hex_str: InitVar[str] = None

    def __post_init__(self, hex_str: str):
        self.__dict__['value'] = int(hex_str, 16)


@dataclass(frozen=True)
class State(Expr):
    init: Optional[Expr] = field(init=False)
    next: Optional[Expr] = field(init=False)


@dataclass(frozen=True)
class BitvecState(Bitvec, State):
    init: Optional[Bitvec]
    next: Optional[Bitvec]


@dataclass(frozen=True)
class ArrayState(Array, State):
    next: Optional[Array]


@dataclass(frozen=True)
class IndexedOperator(Bitvec):
    bitvec: Bitvec


@dataclass(frozen=True)
class ExtensionOperator(IndexedOperator):
    w: int


@dataclass(frozen=True)
class Sext(ExtensionOperator):
    pass


@dataclass(frozen=True)
class Uext(ExtensionOperator):
    pass


@dataclass(frozen=True)
class ExtractionOperator(IndexedOperator):
    upper: int
    lower: int


@dataclass(frozen=True)
class Slice(ExtractionOperator):
    pass


@dataclass(frozen=True)
class UnaryOperator(Expr):
    expr: Expr


@dataclass(frozen=True)
class BitwiseOperator(Bitvec):
    pass


@dataclass(frozen=True)
class UnaryBitwiseOperator(UnaryOperator, BitwiseOperator):
    expr: Bitvec


@dataclass(frozen=True)
class Not(UnaryBitwiseOperator):
    pass


@dataclass(frozen=True)
class ArithmeticOperator(Bitvec):
    pass


@dataclass(frozen=True)
class UnaryArithmeticOperator(UnaryOperator, ArithmeticOperator):
    expr: Bitvec


@dataclass(frozen=True)
class Inc(UnaryArithmeticOperator):
    pass


@dataclass(frozen=True)
class Dec(UnaryArithmeticOperator):
    pass


@dataclass(frozen=True)
class Neg(UnaryArithmeticOperator):
    pass


@dataclass(frozen=True)
class ReductionOperator(UnaryOperator, Bitvec):
    expr: Bitvec


@dataclass(frozen=True)
class Redand(ReductionOperator):
    pass


@dataclass(frozen=True)
class Redor(ReductionOperator):
    pass


@dataclass(frozen=True)
class Redxor(ReductionOperator):
    pass


@dataclass(frozen=True)
class BinaryOperator(Expr):
    expr1: Expr
    expr2: Expr


@dataclass(frozen=True)
class BooleanOperator(BinaryOperator, Bitvec):
    expr1: Bitvec
    expr2: Bitvec


@dataclass(frozen=True)
class Iff(BooleanOperator):
    pass


@dataclass(frozen=True)
class Implies(BooleanOperator):
    pass


@dataclass(frozen=True)
class EqualityOperator(BinaryOperator, Bitvec):
    pass


@dataclass(frozen=True)
class Eq(EqualityOperator):
    pass


@dataclass(frozen=True)
class Neq(EqualityOperator):
    pass


@dataclass(frozen=True)
class InequalityOperator(BinaryOperator, Bitvec):
    expr1: Bitvec
    expr2: Bitvec


@dataclass(frozen=True)
class Sgt(InequalityOperator):
    pass


@dataclass(frozen=True)
class Ugt(InequalityOperator):
    pass


@dataclass(frozen=True)
class Sgte(InequalityOperator):
    pass


@dataclass(frozen=True)
class Ugte(InequalityOperator):
    pass


@dataclass(frozen=True)
class Slt(InequalityOperator):
    pass


@dataclass(frozen=True)
class Ult(InequalityOperator):
    pass


@dataclass(frozen=True)
class Slte(InequalityOperator):
    pass


@dataclass(frozen=True)
class Ulte(InequalityOperator):
    pass


@dataclass(frozen=True)
class BinaryBitwiseOperator(BinaryOperator, BitwiseOperator):
    expr1: Bitvec
    expr2: Bitvec


@dataclass(frozen=True)
class And(BinaryBitwiseOperator):
    pass


@dataclass(frozen=True)
class Nand(BinaryBitwiseOperator):
    pass


@dataclass(frozen=True)
class Nor(BinaryBitwiseOperator):
    pass


@dataclass(frozen=True)
class Or(BinaryBitwiseOperator):
    pass


@dataclass(frozen=True)
class Xnor(BinaryBitwiseOperator):
    pass


@dataclass(frozen=True)
class Xor(BinaryBitwiseOperator):
    pass


@dataclass(frozen=True)
class ShiftOperator(BinaryOperator, Bitvec):
    expr1: Bitvec
    expr2: Bitvec


@dataclass(frozen=True)
class Sll(ShiftOperator):
    pass


@dataclass(frozen=True)
class Sra(ShiftOperator):
    pass


@dataclass(frozen=True)
class Srl(ShiftOperator):
    pass


@dataclass(frozen=True)
class RotateOperator(BinaryOperator, Bitvec):
    expr1: Bitvec
    expr2: Bitvec


@dataclass(frozen=True)
class Rol(RotateOperator):
    pass


@dataclass(frozen=True)
class Ror(RotateOperator):
    pass


@dataclass(frozen=True)
class BinaryArithmeticOperator(BinaryOperator, ArithmeticOperator):
    expr1: Bitvec
    expr2: Bitvec


@dataclass(frozen=True)
class Add(BinaryArithmeticOperator):
    pass


@dataclass(frozen=True)
class Mul(BinaryArithmeticOperator):
    pass


@dataclass(frozen=True)
class Sdiv(BinaryArithmeticOperator):
    pass


@dataclass(frozen=True)
class Udiv(BinaryArithmeticOperator):
    pass


@dataclass(frozen=True)
class Smod(BinaryArithmeticOperator):
    pass


@dataclass(frozen=True)
class Srem(BinaryArithmeticOperator):
    pass


@dataclass(frozen=True)
class Urem(BinaryArithmeticOperator):
    pass


@dataclass(frozen=True)
class Sub(BinaryArithmeticOperator):
    pass


@dataclass(frozen=True)
class OverflowOperator(BinaryOperator, Bitvec):
    expr1: Bitvec
    expr2: Bitvec


@dataclass(frozen=True)
class Saddo(OverflowOperator):
    pass


@dataclass(frozen=True)
class Uaddo(OverflowOperator):
    pass


@dataclass(frozen=True)
class Sdivo(OverflowOperator):
    pass


@dataclass(frozen=True)
class Udivo(OverflowOperator):
    pass


@dataclass(frozen=True)
class Smulo(OverflowOperator):
    pass


@dataclass(frozen=True)
class Umulo(OverflowOperator):
    pass


@dataclass(frozen=True)
class Ssubo(OverflowOperator):
    pass


@dataclass(frozen=True)
class Usubo(OverflowOperator):
    pass


@dataclass(frozen=True)
class ConcatenationOperator(BinaryOperator, Bitvec):
    expr1: Bitvec
    expr2: Bitvec


@dataclass(frozen=True)
class Concat(ConcatenationOperator):
    pass


@dataclass(frozen=True)
class ReadOperator(BinaryOperator):
    expr1: Array
    expr2: Expr


@dataclass(frozen=True)
class Read(ReadOperator):
    pass


@dataclass(frozen=True)
class BitvecRead(ReadOperator, Bitvec):
    pass


@dataclass(frozen=True)
class ArrayRead(ReadOperator, Array):
    pass


@dataclass(frozen=True)
class TernaryOperator(Expr):
    expr1: Expr
    expr2: Expr
    expr3: Expr


@dataclass(frozen=True)
class ConditionalOperator(TernaryOperator):
    expr1: Bitvec


@dataclass(frozen=True)
class Ite(ConditionalOperator):
    pass


@dataclass(frozen=True)
class ArrayIte(Ite, Array):
    expr2: Array
    expr3: Array


@dataclass(frozen=True)
class BitvecIte(Ite, Bitvec):
    expr2: Bitvec
    expr3: Bitvec


@dataclass(frozen=True)
class WriteOperator(TernaryOperator, Array):
    expr1: Array


@dataclass(frozen=True)
class Write(WriteOperator):
    pass


@dataclass(frozen=True)
class Init(Node):
    sort: Sort
    state: State
    expr: Expr


@dataclass(frozen=True)
class BitvecInit(Init):
    sort: BitvecSort
    state: BitvecState
    expr: Bitvec


@dataclass(frozen=True)
class ArrayInit(Node):
    sort: ArraySort
    state: ArrayState
    expr: Expr


@dataclass(frozen=True)
class Next(Node):
    sort: Sort
    state: State
    expr: Expr


@dataclass(frozen=True)
class BitvecNext(Next):
    sort: BitvecSort
    state: BitvecState
    expr: Bitvec


@dataclass(frozen=True)
class ArrayNext(Next):
    sort: ArraySort
    state: ArrayState
    expr: Array


@dataclass(frozen=True)
class Bad(Node):
    bitvec: Bitvec


@dataclass(frozen=True)
class Constraint(Node):
    bitvec: Bitvec


@dataclass(frozen=True)
class Fair(Node):
    bitvec: Bitvec


@dataclass(frozen=True)
class Output(Node):
    expr: Expr


@dataclass(frozen=True)
class Justice(Node):
    n: int
    bitvec: List[Bitvec]
