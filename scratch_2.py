"""Convert 2020 benchmarks."""

import os
import sys

from llvmlite import binding

from main import btor2llvm_test

if __name__ == '__main__':
    binding.initialize_native_target()
    data_layout: str = binding.Target.from_default_triple().create_target_machine().target_data
    triple: str = binding.get_default_triple()

    err_in_paths = []
    out_paths = []

    out_dir = "/home/alan/Documents/hwmcc20/ll/hwmcc20"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sys.setrecursionlimit(10000)
    for root, dirs, files in os.walk("/home/alan/Documents/hwmcc/hwmcc20benchmarks/hwmcc20/btor2"):
        for file in files:
            if file.endswith(".btor2") or file.endswith(".btor"):
                in_path = os.path.join(root, file)
                if '/btor2/array' in root:
                    out_path = os.path.join(out_dir, 'array-' +
                                            file.replace(".btor2", ".ll").replace(".btor", ".ll"))
                else:
                    out_path = os.path.join(out_dir, 'bv-' +
                                            file.replace(".btor2", ".ll").replace(".btor", ".ll"))
                print(os.path.join(root, file))
                try:
                    btor2llvm_test(in_path, out_path)
                    out_paths.append(out_path + '\n')
                except NotImplementedError:
                    err_in_paths.append(in_path + '\n')

    sys.stderr.writelines(err_in_paths)

    with open('libfuzzer_2020.txt', 'w+') as list_file:
        list_file.writelines(out_paths)

    print("Done!")
