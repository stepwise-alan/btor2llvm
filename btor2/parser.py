from typing import Dict, List, Iterable, TypeVar, Sequence, Optional, Type

from btor2.nodes import Bad, Constraint, Fair, Output, Justice, Node, Sort, BitvecSort, ArraySort, \
    Bitvec, Array, BitvecState, ArrayState, BitvecInput, ArrayInput

T = TypeVar('T', bound=Node)


class Line:
    parser: 'Parser'
    tokens: Sequence[str]
    comment: str
    index: int
    applicable: bool

    def __init__(self, parser: 'Parser', tokens: Sequence[str], comment: str):
        self.parser = parser
        self.tokens = tokens
        self.comment = comment
        self.index = 0
        self.applicable = True

    def node(self, table: Dict[int, T]) -> Optional[T]:
        if not self.applicable:
            return None

        node: Optional[T] = table.get(int(self.tokens[self.index]))
        self.index += 1
        self.applicable &= node is not None
        return node

    # def sort(self) -> Optional[Sort]:
    #     return self.node(self.parser.sort_table)
    #
    # def bitvec(self) -> Optional[BitvecSort]:
    #     return self.node(self.parser.sort_table)


class Parser:
    bitvec_sort_table: Dict[int, BitvecSort]
    array_sort_table: Dict[int, ArraySort]
    bitvec_table: Dict[int, Bitvec]
    array_table: Dict[int, Array]
    bitvec_state_table: Dict[int, BitvecState]
    array_state_table: Dict[int, ArrayState]
    bitvec_input_table: Dict[int, BitvecInput]
    array_input_table: Dict[int, ArrayInput]

    bads: List[Bad]
    constraints: List[Constraint]
    fairs: List[Fair]
    outputs: List[Output]
    justices: List[Justice]

    def __init__(self):
        # self.node_table = {}
        # self.sort_table = {}
        # self.bitvec_sort_table = {}
        # self.array_sort_table = {}
        # self.bitvec_table = {}
        # self.array_table = {}
        # self.bitvec_state_table = {}
        # self.array_state_table = {}
        # self.bitvec_input_table = {}
        # self.array_input_table = {}
        self.bads = []
        self.constraints = []
        self.fairs = []
        self.outputs = []
        self.justices = []

    def parse(self, source: Iterable[str]):
        for line in source:
            before: str
            sep: str
            after: str
            before, sep, after = line.partition(';')

            tokens: List[str] = before.split()
            if len(tokens) == 0 or tokens[0].startswith(';'):
                continue

            comment: str = sep + after
            nid: int = int(tokens[0])

            # cls: Type[Node]
            #
            # applicable: bool = True
            # for cls in Parser.cls_table[tokens[1]]:
            #     args: List[Union[int, str, Node]] = []
            #     applicable = True
            #
            #     token: str
            #     f: Field
            #     for token, f in zip(tokens[2:], fields(cls)[3:]):
            #         if issubclass(f.type, Node):
            #             arg_id: int = int(token)
            #             if arg_id not in self.node_table:
            #                 applicable = False
            #                 break
            #             arg_node: Node = self.node_table[arg_id]
            #             if not isinstance(arg_node, f.type):
            #                 applicable = False
            #                 break
            #             args.append(arg_node)
            #         elif f.type is int:
            #             args.append(int(token))
            #         else:
            #             args.append(token)
            #
            #     if applicable:
            #         if len(tokens) == len(args) + 2
            #         symbol: str = tokens[-1] if
            #         self.node_table[nid] = cls(nid, '', comment, *args)
            #         break
            #
            # assert applicable
