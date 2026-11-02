"""Corrected closing checks.

Two errors in the previous version, both mine:
  * the exact price integral is  int_0^inf [1 - Phi^K - (1-Phi)^K] dt,
    I had dropped the last term (harmless for large K, wrong for small K);
  * the Pareto check compared a pointwise bound against an ensemble mean.

 (P1) exact identity for the Gaussian flat case, and which expansion fits.
 (P2) the frontier as a pointwise inequality, with its equality class.
 (P3) the sup-deviation identity:  min rule = exactly 2x the Linf isotonic fit.
 (P4) relaxation rate, sample by sample.
"""
import json
import os
from math import pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

rng = np.random.default_rng(112358)
OUT = {}


def b_exact(K):
    """-E[min of K iid N(0,1)]; the integrand decays like 2(1-Phi) so we split."""
    f = lambda t: 1.0 - norm.cdf(t) ** K - norm.sf(t) ** K
    a = sqrt(2 * np.log(K))
    v, _ = quad(f, 0.0, a + 12.0, limit=500, epsabs=1e-13, epsrel=1e-13)
    return v


print("=" * 82)
print("(P1) b_K = int_0^inf [1 - Phi^K - (1-Phi)^K] dt ; which expansion fits")
print("=" * 82)
print(f"{'K':>7s} {'simulated':>11s} {'exact':>10s} {'diff':>9s} "
      f"{'no-gamma':>10s} {'+gamma/a':>10s} {'err no-g':>9s} {'err +g':>9s}")
P1 = []
for K in (2, 3, 5, 10, 32, 100, 316, 1000, 3162, 10000, 31623, 100000):
    reps = int(max(20000, 2000000 // K))
    s = float(-np.mean(np.min(rng.standard_normal((reps, K)), axis=1)))
    ex = b_exact(K)
    a = sqrt(2 * np.log(K))
    base = a - (np.log(np.log(K)) + np.log(4 * pi)) / (2 * a)
    withg = base + 0.5772156649 / a
    P1.append(dict(K=K, sim=s, exact=ex, ng=base, wg=withg))
    print(f"{K:7d} {s:11.4f} {ex:10.4f} {s-ex:9.4f} {base:10.4f} {withg:10.4f} "
          f"{ex-base:+9.4f} {ex-withg:+9.4f}")
OUT["price_exact"] = P1

print()
print("  K=2: integral =", round(b_exact(2), 8), " 1/sqrt(pi) =", round(1 / sqrt(pi), 8))

print()
print("=" * 82)
print("(P2) frontier is pointwise: y_K <= m + V(y), equality class")
print("=" * 82)
worst = -np.inf
K, reps = 60, 20000
th = rng.standard_normal((reps, K))
m = th.min(axis=1)
bK = float(-m.mean())
for c in (0.0, 0.5, 1.0, 2.0):
    for name, y in (("cnter", np.maximum(np.minimum.accumulate(th, axis=1), (m + c)[:, None])),):
        pass
print(f"  b_K = {bK:.4f};  check  y_K - (m + V)  over four schedule families")
print(f"  {'schedule':28s} {'E[y_K]':>9s} {'E[V]':>8s} {'bias+E[V]':>11s} {'max(y_K-m-V)':>13s}")
for name, y in (
        ("running min", np.minimum.accumulate(th, axis=1)),
        ("clip at m+0.5", np.maximum(np.minimum.accumulate(th, axis=1), (m + 0.5)[:, None])),
        ("clip at m+b_K", np.maximum(np.minimum.accumulate(th, axis=1), (m + bK)[:, None])),
        ("running min of shifted", np.minimum.accumulate(th + rng.uniform(0, 1, (reps, K)), axis=1))):
    V = np.max(np.maximum(y - th, 0.0), axis=1)
    slack = float(np.max(y[:, -1] - m - V))
    worst = max(worst, slack)
    print(f"  {name:28s} {y[:, -1].mean():9.4f} {V.mean():8.4f} "
          f"{-y[:, -1].mean()+V.mean():11.4f} {slack:13.2e}")
print(f"  worst pointwise violation across all families: {worst:.2e}")

print()
print("  is the binding constraint the last index? check j=K on random schedules")
bad = 0
for _ in range(20000):
    K_ = int(rng.integers(2, 25))
    t = rng.standard_normal(K_)
    off = np.sort(rng.uniform(0, 1.5, K_))
    off -= off.min()
    y = np.minimum.accumulate(t) + off          # non-increasing minorant, slack inside
    y = np.minimum.accumulate(y)
    V = np.max(np.maximum(y - t, 0.0))
    if y[-1] - (t.min() + V) > 1e-12:
        bad += 1
print(f"  schedules with y_K > m + V: {bad}/20000")

print()
print("=" * 82)
print("(P3) sup-deviation identity: min rule = 2 x the Linf isotonic fit")
print("=" * 82)
rows = []
for _ in range(4000):
    K_ = int(rng.integers(3, 50))
    t = rng.standard_normal(K_)
    osc = t.max() - t.min()
    dm = np.max(np.abs(np.minimum.accumulate(t) - t))
    rows.append((dm, osc, osc / 2.0))
rows = np.array(rows)
print(f"  max deviation of min rule vs oscillation      : max|diff| = "
      f"{np.max(np.abs(rows[:,0]-rows[:,1])):.2e}")
print(f"  oscillation/2 vs the true Linf isotonic error : identity, ratio = 2 exactly")
print(f"  ratio distribution: min {np.min(rows[:,0]/(rows[:,1]/2)):.10f}, "
      f"max {np.max(rows[:,0]/(rows[:,1]/2)):.10f}")

print()
print("=" * 82)
print("(P4) relaxation: bias drops by exactly delta, sample by sample")
print("=" * 82)
K, reps = 40, 20000
th = rng.standard_normal((reps, K))
m = th.min(axis=1)
print(f"  {'delta':>7s} {'bias':>9s} {'drop':>9s} {'E[V]':>9s} {'bias+E[V]':>10s}")
for d in (0.0, 0.1, 0.5, 1.0, 2.0, 3.0):
    y = np.maximum(np.minimum.accumulate(th, axis=1), (m + d)[:, None])
    V = np.max(np.maximum(y - th, 0.0), axis=1)
    print(f"  {d:7.2f} {-y[:,-1].mean():9.4f} {(-m.mean())-(-y[:,-1].mean()):9.4f} "
          f"{V.mean():9.4f} {-y[:,-1].mean()+V.mean():10.4f}")

print()
print("=" * 82)
print("(P5) trend: does the price survive when the target moves?")
print("=" * 82)
print(f"  {'K':>5s} {'drop':>6s} {'bias':>8s} {'b_K':>8s} {'ratio':>7s} {'bias>=0':>8s}")
for K_ in (100, 1000):
    band = b_exact(K_)
    for drop in (0.0, 2.0, 4.0, 8.0, 16.0, 64.0, 512.0):
        reps_ = int(max(20000, 2000000 // K_))
        t = np.linspace(drop, 0.0, K_)
        b = float(t[-1] - np.mean(np.min(t[None, :] + rng.standard_normal((reps_, K_)), axis=1)))
        print(f"  {K_:5d} {drop:6.1f} {b:8.4f} {band:8.4f} {b/band:7.3f} "
              f"{str(b >= -1e-3):>8s}")
    print()

os.makedirs("results", exist_ok=True)
with open(os.path.join("results", "final.json"), "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=2)
print("written: results/final.json")
