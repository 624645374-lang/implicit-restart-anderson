import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.spatial.distance import cdist

import am_sinkhorn


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "sinkhorn_fig4"
FIGURES = ROOT / "figures" / "sinkhorn_fig4"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def make_problem():
    digits = scipy.io.loadmat(ROOT / "data" / "mnist.mat")["testX"].T
    n = 28
    aa = np.resize(digits[:, 123].astype(np.float64), n**2)
    bb = np.resize(digits[:, 543].astype(np.float64), n**2)
    a, b = aa / aa.sum(), bb / bb.sum()

    index = np.arange(n**2)
    points = np.column_stack((index // n, index % n))
    M = cdist(points, points) ** 2
    M /= np.median(M)

    a[aa == 0], b[bb == 0] = 1e-6, 1e-6
    a, b = a / a.sum(), b / b.sum()
    return a, b, M


def load_error(filename):
    with open(RESULTS / filename, "rb") as file:
        _, error, _ = pickle.load(file)
    return np.asarray(error)


def main():
    a, b, M = make_problem()
    common = dict(
        a=a, b=b, M=M, reg=1e-3, stopThr=1e-12,
        numItermax=2000, beta=1.0, log=True, verbose=False,
    )

    old_directory = Path.cwd()
    os.chdir(RESULTS)
    try:
        print("Running Sinkhorn ...")
        am_sinkhorn.sinkhorn(**common)

        print("Running AM-II(5) ...")
        am_sinkhorn.anderson2_sinkhorn(**common, m=5, type2=True)

        print("Running rAM-II(5) ...")
        am_sinkhorn.am2_sinkhorn(**common, tao=0.0, m=5, type2=True)

        print("Running IRAM-II(4,2) ...")
        am_sinkhorn.iram_sinkhorn(
            **common, tao=1e-16, m=4, n=2, mit=15, type2=True
        )
    finally:
        os.chdir(old_directory)

    curves = {
        "Sinkhorn": load_error("sinkhorn.ser"),
        "AM-II(5)": load_error("anderson2_5.ser"),
        "rAM-II(5)": load_error("am2_5.ser"),
        "IRAM-II(4,2)": load_error("iram2_4_2_15.ser"),
    }
    np.savez(RESULTS / "fig4_errors.npz", **curves)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.semilogy(curves["Sinkhorn"], "--", color="firebrick",
                linewidth=1.5, label="Sinkhorn")
    ax.semilogy(curves["AM-II(5)"], color="darkorange", marker="s",
                markerfacecolor="none", markevery=20, linewidth=1.5,
                markersize=6, label="AM-II(5)")
    ax.semilogy(curves["rAM-II(5)"], color="purple", marker="o",
                markerfacecolor="none", markevery=20, linewidth=1.5,
                markersize=6, label="rAM-II(5)")
    ax.semilogy(curves["IRAM-II(4,2)"], color="blue", marker="d",
                markerfacecolor="none", markevery=20, linewidth=1.5,
                markersize=6, label="IRAM-II(4,2)")

    ax.set_xlim(-50, 1050)
    ax.set_ylim(2e-13, 5)
    ax.set_xticks(np.arange(0, 1001, 200))
    ax.set_xlabel("Iterations", fontsize="xx-large")
    ax.set_ylabel("Function value bound", fontsize="xx-large")
    ax.tick_params(axis="both", which="major", labelsize="x-large")
    ax.legend(loc="lower left", fontsize="x-large", frameon=True)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "sinkhorn_fig4.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "sinkhorn_fig4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", FIGURES / "sinkhorn_fig4.pdf")


if __name__ == "__main__":
    main()
