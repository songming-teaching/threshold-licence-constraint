"""E1 figure: the frontier, drawn as bias against over-trust.

Palette (validated with dataviz/scripts/validate_palette.js, light mode, all pairs):
  slot1 #2a78d6 blue   -> shifted running minimum (on the line)
  slot2 #eb6834 orange -> L2 isotonic (above the line)
  slot3 #1baf7a aqua   -> L_inf isotonic (above the line; relief: direct labels)
Palette passes: lightness band, chroma floor, CVD 9.2, normal-vision 24.0.
Aqua sits at 2.74:1 contrast, so every series is direct-labelled, no legend-only identity.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
DATA = os.path.join(EXP, "data")
FIGS = os.path.join(EXP, "figs")
os.makedirs(FIGS, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

df = pd.read_csv(os.path.join(DATA, "e1_frontier.csv"), comment="#")

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), sharey=True)

for ax, K in zip(axes, (20, 50, 100)):
    d = df[df.K == K]
    line = d[d.rule == "min+delta"].sort_values("overtrust")
    bK = float(line.bK.iloc[0])

    xs = np.linspace(0, line.overtrust.max() * 1.02, 50)
    ax.plot(xs, bK - xs, color=GRID, lw=2.0, zorder=1,
            solid_capstyle="round")
    # put the line label low-left, well away from the delta marker at (bK, 0)
    ax.text(0.30, bK - 1.55, r"bias $= b_K -$ over-trust", color=INK2, fontsize=7.5,
            ha="left", va="center",
            bbox=dict(fc="white", ec="none", pad=1.0), zorder=6)

    ax.plot(line.overtrust, line.bias, "o", ms=6, mfc=BLUE, mec="white", mew=1.2,
            zorder=4, label="running min $+\\,\\delta$")

    for rule, col in (("L2 iso", ORANGE), ("Linf iso", AQUA)):
        c = d[d.rule == rule]
        if len(c) == 0:
            continue
        ax.plot(c.overtrust, c.bias, "s", ms=7, mfc=col, mec="white", mew=1.2,
                zorder=3, label=rule.replace("L2", r"$L_2$").replace("Linf", r"$L_\infty$"))
        for _, r in c.iterrows():
            ax.annotate(rule.replace("L2", r"$L_2$").replace("Linf", r"$L_\infty$"),
                        (r.overtrust, r.bias), textcoords="offset points",
                        xytext=(9, -3), fontsize=7.5, color=col, zorder=5)

    # mark delta = bK on the line
    ax.axvline(bK, color=BLUE, lw=0.8, ls=(0, (3, 3)), alpha=0.55, zorder=2)
    ax.annotate(r"$\delta=b_K$", xy=(bK, 0), xytext=(5, -14),
                textcoords="offset points", fontsize=7.5, color=BLUE, zorder=6)

    ax.axy = None
    ax.set_title(f"$K={K}$", fontsize=9.5, color=INK, pad=6)
    ax.set_xlabel("over-trust  $\\mathrm{E}\\,\\max_j\\,(y_j-\\hat\\theta_j)^+$")
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("bias  $\\theta^\\star_K-\\mathrm{E}\\,y_K$")
axes[0].legend(frameon=False, fontsize=7.5, loc="lower left", handletextpad=0.4,
               borderaxespad=0.2)
fig.tight_layout(pad=0.6)
out = os.path.join(FIGS, "fig_frontier.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("wrote", out)
