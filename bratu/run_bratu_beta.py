import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from test_bratu import Nx, Ny, f, iram


def main():
    root = Path(__file__).resolve().parent
    results_dir = root / "results" / "beta_test"
    results_dir.mkdir(parents=True, exist_ok=True)

    old_directory = Path.cwd()
    os.chdir(results_dir)

    try:
        shape = (Nx + 2, Ny + 2)

        print("Running adaptive-beta IRAM-I...")

        iram(
            f,
            np.zeros(shape),
            tol=1e-6,
            maxit=100000,
            tao=1e-16,
            beta=1.0,
            m=25,
            n=15,
            mit=1000,
            type2=False,
            adaptive_beta=True,
        )

        plt.close("all")

        print("Running adaptive-beta IRAM-II...")

        iram(
            f,
            np.zeros(shape),
            tol=1e-6,
            maxit=100000,
            tao=1e-16,
            beta=1.0,
            m=25,
            n=15,
            mit=1000,
            type2=True,
            adaptive_beta=True,
        )

        plt.close("all")

        print("Adaptive-beta experiment completed.")

    finally:
        os.chdir(old_directory)


if __name__ == "__main__":
    main()