import sys
from typing import Dict, List, Union, Optional

import z3  # type: ignore


def main() -> int:
    with open('output.text') as output:
        env: Dict[int, z3.BitVecRef]
        states: Dict[int, z3.BitVecRef]
        new_states: Dict[int, z3.BitVecRef] = {}
        i: int
        line: str
        for i, line in enumerate(output):
            line_no: int = i + 1
            line = line.strip(' \n')
            front, _, end = line.partition(';')
            if not front:
                func_name: str = line.strip(' ;')
                print('Checking {}'.format(func_name))
                if func_name == 'init' or 'next':
                    if func_name == 'next' and len(states) != len(new_states):
                        sys.stderr.write(
                            'Line {}: {}\n\tNumber of states changed before {}. '
                            'Before: {}, after: {}\n'.format(
                                line_no, line, func_name, len(states), len(new_states)))
                    states = new_states
                    new_states = {}
                else:
                    if len(new_states) != 0:
                        sys.stderr.write('Line {}: {}\n\tState modified in {}\n'.format(
                            line_no, line, func_name))
                env = {}
                continue

            front_tokens: List[str] = front.split()
            nid: int = int(front_tokens[0])
            name: str = front_tokens[1]
            args: List[int] = [int(x) for x in front_tokens[2:]]

            end_tokens: List[str] = end.split()
            width: int = int(end_tokens[-1].strip('()'))
            value: int = int(end_tokens[-2], 16)
            actual: z3.BitVecRef = z3.BitVecVal(value, width)
            expected: Optional[Union[z3.BitVecRef, z3.BoolRef]] = None

            assign: bool = end_tokens[1] == '='

            try:
                if name == 'input':
                    pass
                elif name == 'state':
                    if len(states) != 0 and not assign:
                        expected = states[nid]
                elif name == 'const':
                    expected = z3.BitVecVal(int(front_tokens[3], 2), width)
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
                elif name == 'bad':
                    if end_tokens[0] != 'ret:':
                        sys.stderr.write('Line {}: {}\n\tMissing ret:\n'.format(line_no, line))
                    expected = env[args[0]]
                elif name == 'constraint':
                    if end_tokens[0] != 'ret:':
                        sys.stderr.write('Line {}: {}\n\tMissing ret:\n'.format(line_no, line))
                    expected = env[args[0]]
                elif name == 'init' or name == 'next':
                    if not assign:
                        sys.stderr.write('Line {}: {}\n\tMissing =\n'.format(line_no, line))
                    if int(end_tokens[0]) != args[1]:
                        sys.stderr.write('Line {}: {}\n\tIncorrect NID after =. '
                                         'Expected: {}, but got: {}\n'.format(line_no, line,
                                                                              args[1],
                                                                              int(end_tokens[0])))
                    expected = env[args[2]]
                else:
                    sys.stderr.write(
                        'Line {}: {}\n\tUnknown node type: {}\n'.format(line_no, line, name))
            except KeyError as key_error:
                sys.stderr.write(
                    'Line {}: {}\n\tNID not found: {}\n'.format(line_no, line, key_error.args[0]))

            if expected is not None:
                if isinstance(expected, z3.BoolRef):
                    expected = z3.If(expected, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1))
                if not z3.is_true(z3.simplify(expected == actual)):
                    sys.stderr.write('Line {}: {}\n\tIncorrect value. '
                                     'Expected: {}, but got: {}\n'.format(line_no, line,
                                                                          z3.simplify(expected),
                                                                          actual))

            if assign:
                if name != 'state' and name != 'input' and name != 'init' and name != 'next':
                    sys.stderr.write(
                        'Line {}: {}\n\t{} should not modify state.\n'.format(line_no, line, name))
                if int(end_tokens[0]) in new_states:
                    sys.stderr.write(
                        'Line {}: {}\n\tNID already set: {}\n'.format(line_no, line, end_tokens[0]))
                new_states[int(end_tokens[0])] = actual
            else:
                if nid in env:
                    sys.stderr.write(
                        'Line {}: {}\n\tNID already evaluated: {}\n'.format(line_no, line, nid))
                else:
                    env[nid] = actual

    return 0


if __name__ == '__main__':
    sys.exit(main())
