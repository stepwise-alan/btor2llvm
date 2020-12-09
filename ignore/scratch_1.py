"""Convert 2019 benchmarks."""

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

    sys.setrecursionlimit(10000)
    for root, dirs, files in os.walk("/home/alan/Documents/hwmcc/hwmcc19-single-benchmarks/btor2"):
        for file in files:
            in_path = os.path.join(root, file)
            if file.endswith(".btor2") or file.endswith(".btor"):
                out_dir = root.replace("hwmcc19-single-benchmarks/btor2",
                                       "ll/hwmcc19-single-benchmarks")
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_path = os.path.join(out_dir, file.replace(".btor2", ".ll").replace(
                    ".btor", ".ll"))
                print(os.path.join(root, file))
                try:
                    btor2llvm_test(in_path, out_path)
                    out_paths.append(out_path + '\n')
                except NotImplementedError:
                    err_in_paths.append(in_path + '\n')

    sys.stderr.writelines(err_in_paths)

    with open('libfuzzer_2019.txt', 'w+') as list_file:
        list_file.writelines(out_paths)

    print("Done!")
