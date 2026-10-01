from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "beta_test"
FIGURES = ROOT / "figures" / "beta_test"

FIGURES.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
})


def load_beta(filename):
    with open(RESULTS / filename, "rb") as file:
        beta_values = pickle.load(file)

    return np.asarray(beta_values, dtype=float)


def draw_beta_figure(
    data_filename,
    output_name,
):
    beta_values = load_beta(data_filename)
    iterations = np.arange(len(beta_values))

    # 保存为CSV，方便审稿人查看
    np.savetxt(
        RESULTS / f"{output_name}.csv",
        np.column_stack((iterations, beta_values)),
        delimiter=",",
        header="iteration,beta",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(4.7, 3.5))

    ax.semilogy(
        iterations,
        beta_values,
        color="blue",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Iterations",
        fontsize=15,
    )

    ax.set_ylabel(
        r"$\beta_k$",
        fontsize=15,
    )

    ax.set_ylim(3e-6, 1.7)

    if len(iterations) > 1:
        maximum_iteration = int(iterations[-1])

        ax.set_xlim(
            -0.05 * maximum_iteration,
            1.05 * maximum_iteration,
        )
        # 横轴每隔100次迭代显示一个刻度
        last_tick = (maximum_iteration // 100) * 100

        ax.set_xticks(
            np.arange(0, last_tick + 1, 100)
        )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )

    ax.grid(False, which="both")

    fig.tight_layout()

    fig.savefig(
        FIGURES / f"{output_name}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURES / f"{output_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    draw_beta_figure(
        "iram1_25_15_beta.ser",
        "iram1_beta",
    )

    draw_beta_figure(
        "iram2_25_15_beta.ser",
        "iram2_beta",
    )

    print("Beta figures generated.")


if __name__ == "__main__":
    main()