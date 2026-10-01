import csv
import os
import pickle
import time
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from test_bratu import Nx, Ny, f, iram


# Table 2 的参数
M = 25

N_VALUES = [
    1, 2, 4, 6, 8, 10, 12,
    14, 15, 16, 18, 20, 22, 24,
]

TOL = 1e-6
MAXIT = 1000
TAO = 1e-16
BETA = 6e-6
MIT = 1000


def main():
    root = Path(__file__).resolve().parent
    output_dir = root / "results" / "table2"
    output_dir.mkdir(parents=True, exist_ok=True)

    old_directory = Path.cwd()
    os.chdir(output_dir)

    rows = []

    try:
        methods = [
            ("IRAM-I", False),
            ("IRAM-II", True),
        ]

        for method_name, type2 in methods:
            for n in N_VALUES:
                print(f"Running {method_name}: m={M}, n={n} ...")

                log_name = f"{method_name.lower()}_m{M}_n{n}.log"

                start = time.perf_counter()

                # 每次实验都使用相同的零初值
                with open(log_name, "w", encoding="utf-8") as log_file:
                    with redirect_stdout(log_file):
                        iram(
                            f,
                            np.zeros((Nx + 2, Ny + 2)),
                            tol=TOL,
                            maxit=MAXIT,
                            tao=TAO,
                            beta=BETA,
                            m=M,
                            n=n,
                            mit=MIT,
                            type2=type2,
                        )

                elapsed = time.perf_counter() - start
                plt.close("all")

                if type2:
                    result_filename = f"iram2_{M}_{n}.ser"
                else:
                    result_filename = f"iram1_{M}_{n}.ser"

                with open(result_filename, "rb") as result_file:
                    residual, time_history = pickle.load(result_file)

                iterations = len(residual) - 1
                final_residual = float(residual[-1])
                converged = final_residual <= TOL

                if converged:
                    reported_k = str(iterations)
                else:
                    reported_k = ">10^3"

                rows.append({
                    "method": method_name,
                    "m": M,
                    "n": n,
                    "iterations": iterations,
                    "converged": converged,
                    "reported_k": reported_k,
                    "final_residual": final_residual,
                    "runtime_seconds": elapsed,
                })

                print(
                    f"{method_name}, n={n}: "
                    f"k={reported_k}, "
                    f"final residual={final_residual:.6e}, "
                    f"time={elapsed:.2f}s"
                )

        # 保存详细数据
        detailed_csv = output_dir / "table2_detailed_results.csv"

        with open(
            detailed_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "method",
                    "m",
                    "n",
                    "iterations",
                    "converged",
                    "reported_k",
                    "final_residual",
                    "runtime_seconds",
                ],
            )

            writer.writeheader()
            writer.writerows(rows)

        # 保存与论文表格结构相同的结果
        table_csv = output_dir / "table2_iterations.csv"

        with open(
            table_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(["method"] + N_VALUES)

            for method_name in ["IRAM-I", "IRAM-II"]:
                method_rows = [
                    row for row in rows
                    if row["method"] == method_name
                ]

                values = {
                    row["n"]: row["reported_k"]
                    for row in method_rows
                }

                writer.writerow(
                    [method_name]
                    + [values[n] for n in N_VALUES]
                )

        print("\nTable 2 experiment completed.")
        print(f"Summary: {table_csv}")
        print(f"Detailed results: {detailed_csv}")

    finally:
        os.chdir(old_directory)


if __name__ == "__main__":
    main()