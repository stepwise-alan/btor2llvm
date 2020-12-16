import csv
import os
import statistics
import urllib.request
from typing import List

import pandas as pd


def get_results() -> pd.DataFrame:
    hwmcc20_csv_filename: str = "hwmcc20-combined-all.csv"
    hwmcc20_csv_url: str = "http://fmv.jku.at/hwmcc20/hwmcc20-combined-all.csv"

    if not os.path.exists(hwmcc20_csv_filename):
        urllib.request.urlretrieve(hwmcc20_csv_url, hwmcc20_csv_filename)

    df: pd.DataFrame = pd.DataFrame(
        columns=["index", "result",
                 "sat_count", "uns_count", "unk_count",
                 "low_bound", "high_bound", "avg_bound",
                 "low_real", "high_real", "avg_real",
                 "low_time", "high_time", "avg_time",
                 "low_mem", "high_mem", "avg_mem"])

    with open(hwmcc20_csv_filename, newline="") as in_file:
        reader = csv.reader(in_file, delimiter=";")
        next(reader, None)

        for row in reader:
            statuses: List[str] = [row[i] for i in range(2, len(row), 6)]
            bounds: List[int] = [int(row[i]) for i in range(3, len(row), 6)]
            reals: List[float] = [float(row[i]) for i in range(4, len(row), 6)]
            times: List[float] = [float(row[i]) for i in range(5, len(row), 6)]
            mems: List[float] = [float(row[i]) for i in range(6, len(row), 6)]

            for status in statuses:
                if status not in ('sat', 'uns', 'unk', 'time', 'mem'):
                    print(status)
                assert status in ('sat', 'uns', 'unk', 'time', 'mem')
            for bound in bounds:
                assert isinstance(bound, int)
            for real in reals:
                assert isinstance(real, float)
            for time in times:
                assert isinstance(time, float)
            for mem in mems:
                assert isinstance(mem, float)

            # http://fmv.jku.at/hwmcc20/hwmcc20-combined-all.txt
            #         cnt  ok sat uns dis fld  to mo unk   real    time   space   max best uniq
            #     avr 639 547  66 481   0  92   0  0  92  71456 1049649 2282407 49141  100   23
            #   nuxmv 639 500  50 450   0 139 139  0   0  99664  339693  505974  6898   12    2
            #    pono 639 386  58 328   0 253 219 22  12  63789  288937  596666 77761   38    1
            #   cosa2 639 373  58 315   0 266 252  1  13 136153  537672  585138 14623   17    0
            #  btormc 639 333  59 274   0 306 305  0   1  53822   53797   47550  2565  208    2
            #     abc 324 262  56 206   0  62  62  0   0  36505  429078  344728 31293   29    1
            #   abc17 324 262  55 207   0  62  62  0   0  39465  470045  352773 31820   10    1
            #  pdtrav 324 245  45 200   0  79  76  0   3  74891  429930  713508 30090    8    2
            #     avy 324 236  40 196   0  88  88  0   0  44055  437528  426064 14949   25    0
            #   nmtip 324 210  17 193  26  88  88  0   0  24949   24933   17119  3117  142   13
            # camical 324  43  43   0   0 281 142  0 139  30048   30046    9821   945    8    0
            #
            # Only one model checker (nmtip) seems to be unsound.

            sat_count: int = statuses.count('sat')
            uns_count: int = statuses.count('uns')
            unk_count: int = statuses.count('unk')
            time_count: int = statuses.count('time')

            result: str
            if sat_count > uns_count:
                result = 'sat'
            elif uns_count > sat_count:
                result = 'unsat'
            else:
                result = 'unknown'

            df = df.append({
                "index": row[0],
                "result": result,

                "sat_count": sat_count,
                "uns_count": uns_count,
                "unk_count": unk_count,

                "low_bound": min(bounds),
                "high_bound": max(bounds),
                "avg_bound": statistics.mean(bounds),

                "low_real": min(reals),
                "high_real": max(reals),
                "avg_real": statistics.mean(reals),

                "low_time": min(times),
                "high_time": max(times),
                "avg_time": statistics.mean(times),

                "low_mem": min(mems),
                "high_mem": max(mems),
                "avg_mem": statistics.mean(mems)
            }, ignore_index=True)

    return df


if __name__ == '__main__':
    hwmcc20_results: pd.DataFrame = get_results()
    hwmcc20_results.to_csv("hwmcc20-results.csv", index=False)
