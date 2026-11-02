"""E1: the certificate-bias frontier, delta sweep.

Claims under test
  C1  the shifted running minimum lands exactly on the line bias + overtrust = b_K
      (Corollary 3.3), for every delta and every K
  C2  the L2 and L_inf isotonic fits sit strictly above the line (Corollary 3.4)
  C3  the bias crosses zero at delta = b_K

Outputs
  exp/data/e1_frontier.csv
  exp/data/e1_summary.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog
from scipy.stats import norm

SEED = 20261102
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
DATA = os.path.join(EXP, "data")
os.makedirs(DATA, exist_ok=True)

REPS = 200_000
KS = (20, 50, 100)


def b_exact(K):
    f = lambda t: 1.0 - norm.cdf(t) ** K - norm.sf(t) ** K
    return float(quad(f, 0.0, np.sqrt(2 * np.log(K)) + 14.0, limit=600,
                      epsabs=1e-12, epsrel=1e-12)[0])


def iso_l2(th):
    """Pool-adjacent-violators, least-squares isotonic fit onto non-increasing."""
    vals, wts = [], []
    for t in th:
        vals.append(float(t)); wts.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2)
    return np.array([v for v, w in zip(vals, wts) for _ in range(w)])


def iso_linf(th):
    """L_inf isotonic fit by LP. Only used on small K."""
    n = len(th)
    c = np.zeros(n + 1); c[-1] = 1.0
    A, b = [], []
    for k in range(n):
        r = np.zeros(n + 1); r[k] = 1; r[-1] = -1; A.append(r); b.append(th[k])
        r = np.zeros(n + 1); r[k] = -1; r[-1] = -1; A.append(r); b.append(-th[k])
    for k in range(n - 1):
        r = np.zeros(n + 1); r[k] = -1; r[k + 1] = 1; A.append(r); b.append(0.0)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                  bounds=[(None, None)] * n + [(0, None)], method="highs")
    return res.x[:n]


def stats(y, th):
    """bias and over-trust for a batch, batched over paths."""
    bias = float(-np.mean(y[:, -1]))
    ot = float(np.mean(np.max(np.maximum(y - th, 0.0), axis=1)))
    se_b = float(np.std(-y[:, -1], ddof=1) / np.sqrt(len(y)))
    otv = np.max(np.maximum(y - th, 0.0), axis=1)
    se_o = float(np.std(otv, ddof=1) / np.sqrt(len(y)))
    return bias, ot, se_b, se_o


rows = []
t_start = time.time()

for K in KS:
    rng = np.random.default_rng(SEED + K)
    th = rng.standard_normal((REPS, K))          # flat target, unit variance
    mk = th.min(axis=1)
    bK = float(-mk.mean())

    # -------- (a) shifted running minimum, delta sweep
    for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        d = frac * bK
        y = np.maximum(np.minimum.accumulate(th, axis=1), (mk + d)[:, None])
        bias, ot, sb, so = stats(y, th)
        rows.append(dict(K=K, rule="min+delta", delta=float(d), frac=frac,
                         bias=bias, overtrust=ot, se_bias=sb, se_overtrust=so,
                         total=bias + ot, bK=bK))

    # -------- (b)/(c) isotonic competitors: subsample, they are order-of-magnitude slower
    sub = 3000
    ths = th[:sub]
    y2 = np.array([iso_l2(t) for t in ths])
    yl = np.array([iso_linf(t) for t in ths]) if K <= 60 else None
    for name, yy in (("L2 iso", y2), ("Linf iso", yl)):
        if yy is None:
            continue
        bias, ot, sb, so = stats(yy, ths)
        rows.append(dict(K=K, rule=name, delta=0.0, frac=-1.0,
                         bias=bias, overtrust=ot, se_bias=sb, se_overtrust=so,
                         total=bias + ot, bK=bK))

    print(f"K={K}: bK={bK:.4f}, {time.time()-t_start:.1f}s")

# ------------------------------------------------------------------ checks
print("\n" + "=" * 78)
print("C1  shifted min lies on the line, |total - bK|")
print("=" * 78)
print(f"  {'K':>5s} {'delta':>8s} {'bias':>9s} {'overtrust':>10s} {'total':>9s} "
      f"{'bK':>8s} {'dev':>9s}")
worst_dev = 0.0
for r in rows:
    if r["rule"] != "min+delta":
        continue
    dev = r["total"] - r["bK"]
    worst_dev = max(worst_dev, abs(dev))
    print(f"  {r['K']:5d} {r['delta']:8.4f} {r['bias']:9.4f} {r['overtrust']:10.4f} "
          f"{r['total']:9.4f} {r['bK']:8.4f} {dev:9.5f}")
print(f"  worst deviation from the line: {worst_dev:.2e}")

print()
print("=" * 78)
print("C2  isotonic competitors lie strictly above the line")
print("=" * 78)
print(f"  {'K':>5s} {'rule':>10s} {'bias':>9s} {'overtrust':>10s} {'total':>9s} "
      f"{'excess':>9s} {'z':>7s}")
for r in rows:
    if r["rule"] == "min+delta":
        continue
    exc = r["total"] - r["bK"]
    se = np.hypot(r["se_bias"], r["se_overtrust"])
    print(f"  {r['K']:5d} {r['rule']:>10s} {r['bias']:9.4f} {r['overtrust']:10.4f} "
          f"{r['total']:9.4f} {exc:+9.4f} {exc/se:7.1f}")

print()
print("=" * 78)
print("C3  bias crosses zero at delta = bK")
print("=" * 78)
for K in KS:
    sub = [r for r in rows if r["rule"] == "min+delta" and r["K"] == K]
    sub.sort(key=lambda r: r["delta"])
    for r1, r2 in zip(sub, sub[1:]):
        if r1["bias"] >= 0 >= r2["bias"]:
            # linear interpolation
            w = r1["bias"] / (r1["bias"] - r2["bias"])
            d0 = r1["delta"] + w * (r2["delta"] - r1["delta"])
            print(f"  K={K:4d}: zero crossing at delta={d0:.4f}, bK={r1['bK']:.4f}, "
                  f"rel err={abs(d0-r1['bK'])/r1['bK']*100:.2f}%")

# ------------------------------------------------------------------ write
csv = os.path.join(DATA, "e1_frontier.csv")
with open(csv, "w", encoding="utf-8") as fh:
    fh.write(f"# E1 frontier sweep | generated by exp/code/e1_frontier.py | seed={SEED} "
             f"| reps={REPS} | isotonic subsample=3000\n")
    fh.write("# rule: min+delta = shifted running minimum; L2 iso / Linf iso = isotonic fits\n")
    fh.write("# bias = -E[y_K]; overtrust = E max_j (y_j - thetahat_j)^+; total = bias+overtrust\n")
    fh.write("K,rule,delta,frac,bias,overtrust,se_bias,se_overtrust,total,bK\n")
    for r in rows:
        fh.write(f"{r['K']},{r['rule']},{r['delta']:.6f},{r['frac']:.3f},"
                 f"{r['bias']:.6f},{r['overtrust']:.6f},{r['se_bias']:.6f},"
                 f"{r['se_overtrust']:.6f},{r['total']:.6f},{r['bK']:.6f}\n")

summary = dict(seed=SEED, reps=REPS, Ks=list(KS), worst_line_deviation=worst_dev,
               elapsed_s=round(time.time() - t_start, 2), n_rows=len(rows))
with open(os.path.join(DATA, "e1_summary.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
print(f"\nwrote {csv}")
print(f"elapsed {time.time()-t_start:.1f}s")
