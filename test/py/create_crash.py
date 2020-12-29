from typing import List

if __name__ == '__main__':
    corpus: List[int] = []
    with open('witness.txt') as witness_file:
        for line in witness_file:
            tokens: List[str] = line.split()
            if len(tokens) == 0:
                continue
            if tokens[0] == 'sat' or tokens[0].startswith('b'):
                continue
            if tokens[0].startswith('@'):
                continue
            if tokens[0].startswith('#'):
                continue
            if tokens[0] == '.':
                break
            if tokens[0].isdecimal() and tokens[1].isdecimal():
                b: int = int(tokens[1], 2)
                for _ in range(int((len(tokens[1]) + 7) / 8)):
                    corpus.append(b % (2 ** 8))
                    b = b >> 8

    with open('crash', 'wb+') as crash_file:
        crash_file.write(bytes(corpus))
