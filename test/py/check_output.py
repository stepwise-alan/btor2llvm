import sys
from typing import Dict, List, Union, Optional

import z3


def main() -> int:
    with open('output.text') as output:
        env: Dict[int, z3.BitVecRef]
        i: int
        line: str
        for i, line in enumerate(output):
            line_no: int = i + 1
            line = line.strip(' \n')
            front, _, end = line.partition(';')
            if not front:
                print('Checking {}'.format(line.strip(';')))
                env = dict()
                continue

            tokens: List[str] = front.split()
            nid: int = int(tokens[0])
            name: str = tokens[1]
            args: List[int] = [int(x) for x in tokens[2:]]
            width: int = int(end.split()[0].strip('():'))
            value: int = int(end.split()[1], 16)
            actual: z3.BitVecRef = z3.BitVecVal(value, width)
            expected: Optional[Union[z3.BitVecRef, z3.BoolRef]] = None

            try:
                if name == 'input':
                    pass
                elif name == 'state':
                    pass
                elif name == 'const':
                    pass
                elif name == 'eq':
                    expected = env[args[1]] == env[args[2]]
                elif name == 'neq':
                    expected = env[args[1]] != env[args[2]]
                elif name == 'not':
                    expected = ~env[args[1]]
                elif name == 'and':
                    expected = env[args[1]] & env[args[2]]
                elif name == 'or':
                    expected = env[args[1]] | env[args[2]]
                elif name == 'add':
                    expected = env[args[1]] + env[args[2]]
                elif name == 'concat':
                    expected = z3.Concat(env[args[1]], env[args[2]])
                elif name == 'ite':
                    expected = z3.If(env[args[1]] == z3.BitVecVal(1, 1), env[args[2]], env[args[3]])
                elif name == 'redor':
                    expected = z3.BVRedOr(env[args[1]])
                elif name == 'sext':
                    expected = z3.SignExt(args[2], env[args[1]])
                elif name == 'sgt':
                    expected = env[args[1]] > env[args[2]]
                elif name == 'slice':
                    expected = z3.Extract(args[2], args[3], env[args[1]])
                elif name == 'uext':
                    expected = z3.ZeroExt(args[2], env[args[1]])
                else:
                    sys.stderr.write(
                        'Line {}: {}\n\tUnknown node type: {}\n'.format(line_no, line, name))
            except KeyError as key_error:
                sys.stderr.write(
                    'Line {}: {}\n\tNID not found: {}\n'.format(line_no, line, key_error.args[0]))

            if expected is not None:
                if isinstance(expected, z3.BoolRef):
                    expected = z3.If(expected, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1))
                if z3.is_true(z3.simplify(expected != actual)):
                    sys.stderr.write('Line {}: {}\n\tExpected: {}, but got: {}\n'.format(
                        line_no, line, z3.simplify(expected), actual))

            env[nid] = actual

    return 0


if __name__ == '__main__':
    sys.exit(main())
