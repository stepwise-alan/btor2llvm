import os
import subprocess

import pandas

if __name__ == '__main__':
    timeout: int = 60
    btor_dir: str = '../opt/hwmcc20'
    in_dir: str = '../data/hwmcc20'
    out_dir: str = '../out/hwmcc20'
    hwmcc20_results: pandas.DataFrame = pandas.read_csv('./hwmcc20-results.csv')

    for f in sorted(os.listdir(in_dir), key=lambda x: os.stat(os.path.join(in_dir, x)).st_size):
        in_path: str = os.path.join(in_dir, f)
        print(in_path)

        test_name: str = os.path.splitext(f)[0]
        out_path: str = os.path.join(out_dir, test_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        fuzz_target_path: str = os.path.join(out_path, 'a.out')
        p: subprocess.CompletedProcess = subprocess.run(
            ['clang', '-fsanitize=fuzzer', '-flto', '-fuse-ld=lld',
             in_path, '-o', fuzz_target_path])
        if p.returncode != 0:
            exit(1)

        p = subprocess.run(['./a.out', '-max_total_time={}'.format(timeout)], cwd=out_path,
                           capture_output=True)
        if p.returncode == 77:
            result: pandas.Series = hwmcc20_results.query(
                "index == '{}'".format(test_name))['result']
            assert len(result) == 1
            for r in result:
                if r == 'unsat':
                    print('Simplest unsound benchmark:', in_path)
                    exit(0)
