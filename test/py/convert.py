import sys
import os.path
from typing import List

from main import btor2llvm_test

if __name__ == '__main__':
    sys.setrecursionlimit(10000)

    in_dir: str = '../opt/hwmcc20'
    out_dir: str = '../data/hwmcc20'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    err_in_paths: List[str] = []
    for f in os.listdir(in_dir):
        in_path: str = os.path.join(in_dir, f)
        if os.path.isfile(in_path):
            out_path: str = os.path.join(out_dir,
                                         f.replace('.btor2', '.ll').replace('.btor', '.ll'))
            print(in_path)
            try:
                btor2llvm_test(in_path, out_path)
            except NotImplementedError:
                err_in_paths.append(in_path + '\n')

    sys.stderr.writelines(err_in_paths)
