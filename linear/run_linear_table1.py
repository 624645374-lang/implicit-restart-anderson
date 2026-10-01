"""Generate Table 1 for the nonsymmetric linear system."""

import contextlib
import csv
import os
import pickle
from pathlib import Path

import numpy as np

from run_linear_fig1 import problem, suppress_iram_eigenvalue_figure


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results" / "linear_table1"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TOL = 1.0e-6
MAXIT = 1001  # 检查到 k=1000；仍未达到精度时记为 >10^3
BETA = 1.0e-3


def read_residuals(filename):
    with open(filename, "rb") as file:
        return np.asarray(pickle.load(file), dtype=float).reshape(-1)


def iteration_number(residuals):
    for k, residual in enumerate(residuals, start=1):
        if residual <= TOL:
            return k
    return r">10^3"


def run_method(name, filename, function, iram=False):
    print(f"Running {name} ...")
    with open(f"{name}.log", "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file):
            if iram:
                with suppress_iram_eigenvalue_figure():
                    function()
            else:
                function()
    return iteration_number(read_residuals(filename))


def main():
    old_directory = Path.cwd()
    os.chdir(RESULT_DIR)
    try:
        x0 = np.zeros((problem.D_out, 1), dtype=float)

        experiments = [
            (
                "AM-I(inf)",
                "anderson1_1000.ser",
                lambda: problem.anderson1(
                    problem.misfit_total, x0.copy(), TOL, MAXIT, BETA, 1000
                ),
                False,
            ),
            (
                "AM-II(inf)",
                "anderson2_1000.ser",
                lambda: problem.anderson2(
                    problem.misfit_total, x0.copy(), TOL, MAXIT, BETA, 1000
                ),
                False,
            ),
            (
                "AM-I(26)",
                "anderson1_26.ser",
                lambda: problem.anderson1(
                    problem.misfit_total, x0.copy(), TOL, MAXIT, BETA, 26
                ),
                False,
            ),
            (
                "AM-II(26)",
                "anderson2_26.ser",
                lambda: problem.anderson2(
                    problem.misfit_total, x0.copy(), TOL, MAXIT, BETA, 26
                ),
                False,
            ),
            (
                "rAM-I(26)",
                "am1_26.ser",
                lambda: problem.am1(
                    problem.misfit_total, x0.copy(), TOL, MAXIT, 1.0e-16, BETA, 26
                ),
                False,
            ),
            (
                "rAM-II(26)",
                "am2_26.ser",
                lambda: problem.am2(
                    problem.misfit_total, x0.copy(), TOL, MAXIT, 1.0e-16, BETA, 26
                ),
                False,
            ),
            (
                "IRAM-I",
                "iram1.ser",
                lambda: problem.iram(
                    problem.misfit_total,
                    x0.copy(),
                    TOL,
                    MAXIT,
                    0.0,
                    BETA,
                    25,
                    15,
                    1000,
                    False,
                ),
                True,
            ),
            (
                "IRAM-II",
                "iram2.ser",
                lambda: problem.iram(
                    problem.misfit_total,
                    x0.copy(),
                    TOL,
                    MAXIT,
                    0.0,
                    BETA,
                    25,
                    15,
                    1000,
                    True,
                ),
                True,
            ),
        ]

        results = []
        for name, filename, function, is_iram in experiments:
            results.append(run_method(name, filename, function, is_iram))

        with open("linear_table1.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Method", *[item[0] for item in experiments]])
            writer.writerow(["k", *results])

        print("\nMethod          k")
        for experiment, result in zip(experiments, results):
            print(f"{experiment[0]:13s} {result}")
        print(f"\nSaved to: {RESULT_DIR / 'linear_table1.csv'}")
    finally:
        os.chdir(old_directory)


if __name__ == "__main__":
    main()
