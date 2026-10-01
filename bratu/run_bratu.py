import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

from test_bratu import (
    Nx,
    Ny,
    f,
    anderson1,
    anderson2,
    iram,
)


def main():
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    old_directory = Path.cwd()
    os.chdir(results_dir)

    try:
        shape = (Nx + 2, Ny + 2)

        print("Running Anderson-I(26)...")
        anderson1(
            f,
            np.zeros(shape),
            tol=1e-6,
            maxit=100000,
            beta=6e-6,
            m=26,
        )

        print("Running Anderson-II(26)...")
        anderson2(
            f,
            np.zeros(shape),
            tol=1e-6,
            maxit=100000,
            beta=6e-6,
            m=26,
        )

        print("Running IRAM-I(25,15)...")
        iram(
            f,
            np.zeros(shape),
            tol=1e-6,
            maxit=100000,
            tao=1e-16,
            beta=6e-6,
            m=25,
            n=15,
            mit=1000,
            type2=False,
        )

        print("Running IRAM-II(25,15)...")
        iram(
            f,
            np.zeros(shape),
            tol=1e-6,
            maxit=100000,
            tao=1e-16,
            beta=6e-6,
            m=25,
            n=15,
            mit=1000,
            type2=True,
        )

    finally:
        os.chdir(old_directory)


if __name__ == "__main__":
    main()