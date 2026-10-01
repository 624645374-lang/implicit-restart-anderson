import csv
import os
import pickle
from pathlib import Path

import numpy as np

import am_sinkhorn
from run_sinkhorn_fig4 import make_problem


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "sinkhorn_table4"
RESULTS.mkdir(parents=True, exist_ok=True)

M_VALUES = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 50, 100]
TOL = 1e-12


def first_converged_iteration(filename):
    with open(filename, "rb") as file:
        _, errors, _ = pickle.load(file)
    for k, error in enumerate(errors):
        if error <= TOL:
            return k
    return ">10^3"


def main():
    a, b, cost_matrix = make_problem()
    old_directory = Path.cwd()
    os.chdir(RESULTS)

    try:
        iterations = []
        for restart_interval in M_VALUES:
            print(f"Running M = {restart_interval} ...")
            am_sinkhorn.iram_sinkhorn(
                a,
                b,
                cost_matrix,
                reg=1e-3,
                stopThr=TOL,
                numItermax=2000,
                beta=1.0,
                log=True,
                verbose=False,
                tao=1e-16,
                m=4,
                n=2,
                mit=restart_interval,
                type2=True,
            )
            filename = f"iram2_4_2_{restart_interval}.ser"
            iterations.append(first_converged_iteration(filename))

        with open("sinkhorn_table4.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["M", *M_VALUES])
            writer.writerow(["k", *iterations])

        print("\nM:", M_VALUES)
        print("k:", iterations)
        print("Saved:", RESULTS / "sinkhorn_table4.csv")
    finally:
        os.chdir(old_directory)


if __name__ == "__main__":
    main()
