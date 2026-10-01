from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def load_residual(filename):
    with open(RESULTS / filename, "rb") as file:
        data = pickle.load(file)

    # AM-I/II 保存的是 res_v；
    # IRAM-I/II 保存的是 (res_v, time_v)。
    if isinstance(data, tuple):
        data = data[0]

    return np.asarray(data, dtype=float)


curves = [
    ("anderson1_26.ser", "AM-I(26)", "#707070", "o"),
    ("anderson2_26.ser", "AM-II(26)", "#ff7f0e", "s"),
    ("iram1_25_15.ser", "IRAM-I(25,15)", "#e51c47", "^"),
    ("iram2_25_15.ser", "IRAM-II(25,15)", "#0000ff", "d"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})


fig, ax = plt.subplots(figsize=(6.4, 4.7))


for filename, label, color, marker in curves:
    residual = load_residual(filename)
    iterations = np.arange(len(residual))

    # 保存为通用CSV数据
    csv_filename = Path(filename).with_suffix(".csv")
    np.savetxt(
        RESULTS / csv_filename,
        np.column_stack((iterations, residual)),
        delimiter=",",
        header="iteration,residual",
        comments="",
    )

    ax.semilogy(
        iterations,
        residual,
        color=color,
        linewidth=1.5,
        marker=marker,
        markersize=5.5,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=1.0,
        markevery=15,
        label=label,
    )


ax.set_xlabel("Iterations", fontsize=20)
ax.set_ylabel("Residual norm", fontsize=20)

# 与原图一致的横轴范围和刻度
ax.set_xlim(-75, 1600)
ax.set_xticks(np.arange(0, 1501, 250))

# 与原图一致的对数纵轴范围和主刻度
ax.set_ylim(3e-7, 6e4)
ax.set_yticks([1e-5, 1e-3, 1e-1, 1e1, 1e3])

ax.tick_params(axis="both", which="major", labelsize=15)
ax.grid(False, which="both")

ax.legend(
    loc="upper right",
    fontsize=16,
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
)

fig.tight_layout()

fig.savefig(
    FIGURES / "Fig3_bratu.pdf",
    format="pdf",
    bbox_inches="tight",
)

fig.savefig(
    FIGURES / "Fig3_bratu.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()