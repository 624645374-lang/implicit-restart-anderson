import csv
import gc
import os
import pickle
import time
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from test_bratu import (
    Nx,
    Ny,
    f,
    anderson1,
    anderson2,
    am1,
    am2,
    iram,
)


# ============================================================
# Table 3 parameters
# ============================================================

M_VALUES = [20, 25, 30, 35, 40, 45, 50]

# 用 m=1000 表示无限记忆。
# 表中的算法在约 500 次迭代内收敛，因此不会真正达到 m=1000。
M_INFINITY = 1000

N_IRAM = 16

TOL = 1e-6
MAXIT = 2000
BETA = 6e-6

# rAM 的重启阈值
RAM_TAO = 1e-32

# IRAM 的重启阈值
IRAM_TAO = 1e-16
IRAM_MIT = 1000

# 如果程序中途停止，重新运行时跳过已经完成的实验
SKIP_EXISTING = True


def result_filename(method, m):
    """返回 test_bratu.py 自动生成的结果文件名。"""

    if method == "AM-I":
        return f"anderson1_{m}.ser"

    if method == "AM-II":
        return f"anderson2_{m}.ser"

    if method == "rAM-I":
        return f"am1_{m}.ser"

    if method == "rAM-II":
        return f"am2_{m}.ser"

    if method == "IRAM-I":
        return f"iram1_{m}_{N_IRAM}.ser"

    if method == "IRAM-II":
        return f"iram2_{m}_{N_IRAM}.ser"

    raise ValueError(f"Unknown method: {method}")


def run_method(method, m):
    """从相同的零初值运行一种方法。"""

    initial_value = np.zeros((Nx + 2, Ny + 2))

    if method == "AM-I":
        anderson1(
            f,
            initial_value,
            tol=TOL,
            maxit=MAXIT,
            beta=BETA,
            m=m,
        )

    elif method == "AM-II":
        anderson2(
            f,
            initial_value,
            tol=TOL,
            maxit=MAXIT,
            beta=BETA,
            m=m,
        )

    elif method == "rAM-I":
        am1(
            f,
            initial_value,
            tol=TOL,
            maxit=MAXIT,
            tao=RAM_TAO,
            beta=BETA,
            m=m,
        )

    elif method == "rAM-II":
        am2(
            f,
            initial_value,
            tol=TOL,
            maxit=MAXIT,
            tao=RAM_TAO,
            beta=BETA,
            m=m,
        )

    elif method == "IRAM-I":
        iram(
            f,
            initial_value,
            tol=TOL,
            maxit=MAXIT,
            tao=IRAM_TAO,
            beta=BETA,
            m=m,
            n=N_IRAM,
            mit=IRAM_MIT,
            type2=False,
        )

    elif method == "IRAM-II":
        iram(
            f,
            initial_value,
            tol=TOL,
            maxit=MAXIT,
            tao=IRAM_TAO,
            beta=BETA,
            m=m,
            n=N_IRAM,
            mit=IRAM_MIT,
            type2=True,
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def load_residual(filename):
    """读取不同函数保存的残差数据。"""

    with open(filename, "rb") as result_file:
        data = pickle.load(result_file)

    # Anderson 和 IRAM 保存 (res_v, time_v)；
    # rAM 保存 res_v。
    if isinstance(data, tuple):
        residual = data[0]
    else:
        residual = data

    return np.asarray(residual, dtype=float)


def main():
    root = Path(__file__).resolve().parent
    output_dir = root / "results" / "table3"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        "AM-I",
        "AM-II",
        "rAM-I",
        "rAM-II",
        "IRAM-I",
        "IRAM-II",
    ]

    # 最后一项对应表中的无穷大
    experiments = [(m, str(m)) for m in M_VALUES]
    experiments.append((M_INFINITY, "infinity"))

    rows = []

    old_directory = Path.cwd()
    os.chdir(output_dir)

    try:
        for method in methods:
            for actual_m, displayed_m in experiments:
                filename = result_filename(method, actual_m)
                result_path = output_dir / filename

                print(
                    f"Processing {method}, "
                    f"m={displayed_m} "
                    f"(actual m={actual_m})..."
                )

                elapsed = None

                if SKIP_EXISTING and result_path.exists():
                    print(f"Using existing result: {filename}")

                else:
                    log_filename = (
                        f"{method.replace('-', '_').lower()}"
                        f"_m{displayed_m}.log"
                    )

                    start = time.perf_counter()

                    # test_bratu.py 每次迭代都会打印残差，
                    # 将这些输出保存到单独日志中。
                    with open(
                        log_filename,
                        "w",
                        encoding="utf-8",
                    ) as log_file:
                        with redirect_stdout(log_file):
                            run_method(method, actual_m)

                    elapsed = time.perf_counter() - start

                plt.close("all")
                gc.collect()

                residual = load_residual(result_path)

                iterations = len(residual) - 1
                final_residual = float(residual[-1])
                converged = final_residual <= TOL

                if converged:
                    reported_k = str(iterations)
                else:
                    reported_k = f">{MAXIT}"

                rows.append({
                    "method": method,
                    "displayed_m": displayed_m,
                    "actual_m": actual_m,
                    "iterations": iterations,
                    "converged": converged,
                    "reported_k": reported_k,
                    "final_residual": final_residual,
                    "runtime_seconds": (
                        "" if elapsed is None else elapsed
                    ),
                })

                print(
                    f"  k={reported_k}, "
                    f"final residual={final_residual:.6e}"
                )

        # ====================================================
        # 详细实验记录
        # ====================================================

        detailed_csv = output_dir / "table3_detailed_results.csv"

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
                    "displayed_m",
                    "actual_m",
                    "iterations",
                    "converged",
                    "reported_k",
                    "final_residual",
                    "runtime_seconds",
                ],
            )

            writer.writeheader()
            writer.writerows(rows)

        # ====================================================
        # 与论文 Table 3 相同结构的结果
        # ====================================================

        table_csv = output_dir / "table3_iterations.csv"

        column_labels = [
            str(m) for m in M_VALUES
        ] + ["infinity"]

        with open(
            table_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(["method"] + column_labels)

            for method in methods:
                method_rows = [
                    row for row in rows
                    if row["method"] == method
                ]

                values = {
                    row["displayed_m"]: row["reported_k"]
                    for row in method_rows
                }

                writer.writerow(
                    [method]
                    + [values[label] for label in column_labels]
                )

        print("\nTable 3 experiment completed.")
        print(f"Table: {table_csv}")
        print(f"Details: {detailed_csv}")

    finally:
        os.chdir(old_directory)


if __name__ == "__main__":
    main()