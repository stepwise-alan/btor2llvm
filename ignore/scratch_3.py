"""Convert beem_btor benchmarks."""

import os
import sys

from chctools.btor2 import btor2chc

if __name__ == '__main__':
    err_files = []
    for root, dirs, files in os.walk("/home/alan/Documents/hwmcc/beem_btor/btor2"):
        for file in files:
            if file.endswith(".btor2") or file.endswith(".btor"):
                with open(os.path.join(root, file)) as input_file:
                    out_path = root.replace("beem_btor/btor2",
                                            "smt2/beem_btor")
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    with open(os.path.join(out_path, file.replace(".btor2", ".smt2")
                            .replace(".btor", ".smt2")), "w+") as output_file:
                        try:
                            print(root + "/" + file)
                            btor2chc(input_file, output_file, fmt="smt2")
                        except Exception as e:
                            sys.stderr.write(input_file.name + "\n")
                            err_files.append(input_file.name)
                            print(e.with_traceback(None))

    print(err_files)
