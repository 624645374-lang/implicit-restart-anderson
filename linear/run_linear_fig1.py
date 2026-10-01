
from __future__ import annotations

import contextlib
import csv
import os
import pickle
import time
from pathlib import Path

import matplotlib

# Use a non-interactive backend so that the experiment also runs without a GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

try:
    import simple_linear3 as problem
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Please place run_linear_fig1.py and simple_linear3.py in the same folder."
    ) from error


# ---------------------------------------------------------------------------
# Parameters used for the four curves in Fig. 1
# ---------------------------------------------------------------------------
TOL = 1.0e-6
MAXIT = problem.D_out + 1000  # 2000 for the supplied D_out = 1000
BETA = 1.0e-3

AM_M = 26
IRAM_M = 25
IRAM_N = 15
IRAM_TAO = 0.0
IRAM_MIT = 1000

RESULT_DIR = BASE_DIR / "results" / "linear_fig1"
FIGURE_DIR = BASE_DIR / "figures" / "linear_fig1"


@contextlib.contextmanager
def working_directory(path: Path):
    """Temporarily write the .ser files produced by the original code to path."""

    old_directory = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_directory)


@contextlib.contextmanager
def suppress_iram_eigenvalue_figure():
    """Disable only the extra eigenvalue plot at the end of problem.iram().

    The supplied IRAM routine always creates an unrelated eigenvalue figure
    after it has completed the residual experiment.  Suppressing that block
    here leaves every IRAM iteration unchanged and avoids a PGF/LaTeX error on
    computers without a LaTeX installation.
    """

    # Initialize the Agg canvas before temporarily replacing switch_backend.
    temporary_figure = plt.figure()
    plt.close(temporary_figure)

    original_switch_backend = plt.switch_backend
    original_savefig = plt.savefig
    original_eigvals = np.linalg.eigvals

    def diagnostic_eigvals(matrix):
        # A is upper triangular, so its full eigenvalues are its diagonal.
        # This branch is reached only by the post-processing block of iram().
        if matrix is problem.A:
            return np.diag(matrix).copy()
        return original_eigvals(matrix)

    plt.switch_backend = lambda *args, **kwargs: None
    plt.savefig = lambda *args, **kwargs: None
    np.linalg.eigvals = diagnostic_eigvals

    try:
        yield
    finally:
        np.linalg.eigvals = original_eigvals
        plt.savefig = original_savefig
        plt.switch_backend = original_switch_backend
        plt.close("all")


def load_residuals(filename: str) -> np.ndarray:
    """Load one residual history saved by simple_linear3.py."""

    with (RESULT_DIR / filename).open("rb") as file:
        residuals = pickle.load(file)
    return np.asarray(residuals, dtype=float).reshape(-1)


def run_with_log(name: str, function) -> float:
    """Run one method, save its printed iterations, and return elapsed time."""

    log_path = RESULT_DIR / f"{name}.log"
    print(f"Running {name} ...")
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file):
            function()
    elapsed = time.perf_counter() - start
    print(f"  finished in {elapsed:.3f} seconds")
    return elapsed


def run_experiments():
    """Run AM-I, AM-II, IRAM-I, and IRAM-II from the same zero initial value."""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    elapsed_times = {}
    zero_initial_value = np.zeros((problem.D_out, 1), dtype=float)

    with working_directory(RESULT_DIR):
        elapsed_times["AM-I(26)"] = run_with_log(
            "am1_m26",
            lambda: problem.anderson1(
                problem.misfit_total,
                zero_initial_value.copy(),
                tol=TOL,
                maxit=MAXIT,
                beta=BETA,
                m=AM_M,
            ),
        )

        elapsed_times["AM-II(26)"] = run_with_log(
            "am2_m26",
            lambda: problem.anderson2(
                problem.misfit_total,
                zero_initial_value.copy(),
                tol=TOL,
                maxit=MAXIT,
                beta=BETA,
                m=AM_M,
            ),
        )

        with suppress_iram_eigenvalue_figure():
            elapsed_times["IRAM-I(25,15)"] = run_with_log(
                "iram1_m25_n15",
                lambda: problem.iram(
                    problem.misfit_total,
                    zero_initial_value.copy(),
                    tol=TOL,
                    maxit=MAXIT,
                    tao=IRAM_TAO,
                    beta=BETA,
                    m=IRAM_M,
                    n=IRAM_N,
                    mit=IRAM_MIT,
                    type2=False,
                ),
            )

        with suppress_iram_eigenvalue_figure():
            elapsed_times["IRAM-II(25,15)"] = run_with_log(
                "iram2_m25_n15",
                lambda: problem.iram(
                    problem.misfit_total,
                    zero_initial_value.copy(),
                    tol=TOL,
                    maxit=MAXIT,
                    tao=IRAM_TAO,
                    beta=BETA,
                    m=IRAM_M,
                    n=IRAM_N,
                    mit=IRAM_MIT,
                    type2=True,
                ),
            )

    residuals = {
        "AM-I(26)": load_residuals("anderson1_26.ser"),
        "AM-II(26)": load_residuals("anderson2_26.ser"),
        "IRAM-I(25,15)": load_residuals("iram1.ser"),
        "IRAM-II(25,15)": load_residuals("iram2.ser"),
    }
    return residuals, elapsed_times


def save_tables(residuals, elapsed_times):
    """Save residual histories and a concise reproducibility summary."""

    max_length = max(len(values) for values in residuals.values())
    csv_path = RESULT_DIR / "linear_fig1_residuals.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["iteration", *residuals.keys()])
        for iteration in range(max_length):
            row = [iteration]
            for values in residuals.values():
                row.append(values[iteration] if iteration < len(values) else "")
            writer.writerow(row)

    summary_path = RESULT_DIR / "linear_fig1_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["method", "stored_residuals", "final_relative_residual", "time_seconds"])
        for method, values in residuals.items():
            writer.writerow(
                [method, len(values), f"{values[-1]:.16e}", f"{elapsed_times[method]:.9f}"]
            )


def plot_figure(residuals):
    """Plot the four curves using the visual settings in the supplied figure."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.0,
        }
    )

    styles = {
        "AM-I(26)": dict(color="#6f6f6f", marker="o"),
        "AM-II(26)": dict(color="#ff7f00", marker="s"),
        "IRAM-I(25,15)": dict(color="#e51c3a", marker="^"),
        "IRAM-II(25,15)": dict(color="#0000ff", marker="d"),
    }

    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    for label, values in residuals.items():
        iterations = np.arange(len(values))
        axis.semilogy(
            iterations,
            values,
            label=label,
            linewidth=1.5,
            markersize=6.0,
            markerfacecolor="none",
            markeredgewidth=1.15,
            markevery=10,
            **styles[label],
        )

    axis.set_xlim(-40, 830)
    axis.set_ylim(4.0e-7, 6.0)
    axis.set_xticks(np.arange(0, 801, 200))
    axis.set_xlabel("Iterations", fontsize="xx-large")
    axis.set_ylabel("Relative residual norm", fontsize="xx-large")
    axis.tick_params(axis="both", which="major", labelsize="x-large", width=1.0)
    axis.tick_params(axis="both", which="minor", width=0.8)
    axis.grid(False)
    axis.legend(loc="upper right", fontsize="x-large", frameon=True)

    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "linear_fig1.pdf", format="pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / "linear_fig1.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    residuals, elapsed_times = run_experiments()
    save_tables(residuals, elapsed_times)
    plot_figure(residuals)

    print("\nCompleted.")
    for method, values in residuals.items():
        print(
            f"{method:17s}: {len(values):4d} stored residuals, "
            f"final residual = {values[-1]:.6e}"
        )
    print(f"Results: {RESULT_DIR}")
    print(f"Figure : {FIGURE_DIR / 'linear_fig1.pdf'}")


if __name__ == "__main__":
    main()
