"""E2 figure: the price law, four panels.

Palette (validated, light mode, all pairs):
  slot1 #2a78d6 blue, slot2 #eb6834 orange, slot3 #1baf7a aqua.
Aqua at 2.74:1 contrast -> every series is direct-labelled.
No dual axis anywhere; each panel has one y scale.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
FIGS = os.path.join(EXP, "figs")
os.makedirs(FIGS, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

df = pd.read_csv(os.path.join(EXP, "data", "e2_price.csv"), comment="#")
g = df[df.panel == "gauss"].sort_values("K")
t = df[df.panel == "tail"].sort_values("K")
m = df[df.panel == "multi"].sort_values(["d", "K"])

fig, axs = plt.subplots(2, 2, figsize=(9.4, 6.4))
(a, b), (c, d) = axs

# ---------------------------------------------------------------- (a)
a.plot(g.K, g["exact"], "-", color=BLUE, lw=2.0, label="exact (quadrature)")
a.plot(g.K, g.sqrt2, "--", color=ORANGE, lw=1.8, label=r"$\sqrt{2\ln K}$")
a.plot(g.K, g["mod"], "-.", color=AQUA, lw=1.8, label=r"$b^{\rm c}_K+\gamma/a_K$")
a.set_xscale("log")
a.set_xlabel("$K$")
a.set_ylabel("price  $b_K/\\sigma$")
a.set_title("(a) Gaussian: the quadrature and two closed forms", loc="left",
            fontsize=9, color=INK)
a.legend(frameon=False, fontsize=7.5, loc="upper left", handlelength=2.2)
a.annotate(r"$+40\%$ at $K{=}10$", xy=(10, 2.146), xytext=(14, 2.85),
           fontsize=7, color=ORANGE,
           arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7))
a.set_ylim(0, 5.3)

# ---------------------------------------------------------------- (b)
cols = {3: BLUE, 4: ORANGE, 6: AQUA}
for nu, col in cols.items():
    s = t[t.nu == nu]
    b.plot(s.K, s["exact"], "-", color=col, lw=2.0)
    b.plot(s.K, s["exact"], "o", color=col, ms=4.2, mfc=col, mec="white", mew=0.8)
b.set_xscale("log")
b.set_xlabel("$K$")
b.set_ylabel("price  $b_K$")
b.set_title(r"(b) heavy tails: $t_\nu$, exact price", loc="left",
            fontsize=9, color=INK)
for nu, col, y0 in ((3, BLUE, 34), (4, ORANGE, 24), (6, AQUA, 16)):
    b.annotate(rf"$\nu={nu}$", xy=(1.25e5, y0), fontsize=8, color=col, va="center")
b.set_ylim(0, 42)

# ---------------------------------------------------------------- (c)
for nu, col in cols.items():
    s = t[t.nu == nu]
    c.plot(s.K, s.ratio, "-o", color=col, ms=4.5, mfc=col, mec="white", mew=0.9, lw=1.8)
c.axhline(1.0, color=GRID, lw=1.0, zorder=0)
c.set_xscale("log")
c.set_xlabel("$K$")
c.set_ylabel("exact / asymptotic")
c.set_title(r"(c) the Fréchet constant converges from below", loc="left",
            fontsize=9, color=INK)
c.set_ylim(0.82, 1.02)
c.set_xlim(70, 3.2e5)
for nu, col, dy in ((3, BLUE, 5), (4, ORANGE, -9), (6, AQUA, 4)):
    s = t[t.nu == nu]
    c.annotate(rf"$\nu={nu}$", xy=(2.2e5, s.ratio.iloc[-1]),
               xytext=(8, dy), textcoords="offset points", fontsize=8, color=col,
               va="center")

# ---------------------------------------------------------------- (d)
mk = {1: "o", 10: "s", 100: "^"}
m = m.assign(dK=m.d * m.K)
for dd, col in ((1, BLUE), (10, ORANGE), (100, AQUA)):
    s = m[m.d == dd]
    d.plot(s.dK, s.MC, mk[dd], color=col, ms=8, mfc=col, mec="white", mew=1.1,
           label=rf"$d={dd}$")
xs = np.logspace(1.95, 5.15, 120)
d.plot(xs, np.sqrt(2 * np.log(xs)) - (np.log(np.log(xs)) + np.log(4 * np.pi))
       / (2 * np.sqrt(2 * np.log(xs))), "-", color=INK2, lw=1.4, alpha=0.85)
d.annotate("exact $b_{dK}$", xy=(2.3e4, 3.94), xytext=(0, 12),
           textcoords="offset points", fontsize=7.5, color=INK2)
d.set_xscale("log")
d.set_xlabel(r"$dK$  (total comparisons)")
d.set_ylabel("price")
d.set_title("(d) the cost tracks $dK$, not $K$", loc="left", fontsize=9, color=INK)
d.legend(frameon=False, fontsize=7.5, loc="upper left", handletextpad=0.3,
         title="measured at", title_fontsize=7.5)
d.set_ylim(2.2, 4.7)

for ax in axs.ravel():
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(pad=0.7, h_pad=1.6, w_pad=1.6)
out = os.path.join(FIGS, "fig_price.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("wrote", out)
