"""Closing the remaining gaps before writing.

 (G1) the sup-deviation identity, proved quantity checked against the LP:
      Linf isotonic error = (1/2) max drawup = (1/2) min-rule deviation.
 (G2) the lower bound for a monotone (decreasing) target, corrected integral.
 (G3) correlated estimates: the price is set by the decorrelated block length.
 (G4) contamination: MSE of the min rule vs the tempered family.
 (G5) the min rule is the greatest CAUSAL feasible schedule (tested by attack).
"""
import json
import os
from math import pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog
from scipy.stats import norm

rng = np.random.default_rng(60601)
OUT = {}


def b_exact(K):
    f = lambda t: 1.0 - norm.cdf(t) ** K - norm.sf(t) ** K
    v, _ = quad(f, 0.0, sqrt(2 * np.log(K)) + 14.0, limit=600,
                epsabs=1e-14, epsrel=1e-14)
    return v


def iso_linf_err(th):
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
    return float(res.fun)


print("=" * 80)
print("(G1) Linf isotonic error = half the max drawup = half the min-rule deviation")
print("=" * 80)
errs, pred = [], []
for _ in range(600):
    K = int(rng.integers(3, 26))
    t = rng.standard_normal(K)
    draw = float(np.max(np.maximum.accumulate(t) - t))
    errs.append(iso_linf_err(t)); pred.append(0.5 * draw)
errs, pred = np.array(errs), np.array(pred)
print(f"  LP error vs drawup/2 over 600 sequences: "
      f"max|diff| = {np.max(np.abs(errs-pred)):.2e}, "
      f"max rel = {np.max(np.abs(errs-pred)/np.maximum(pred,1e-9)):.2e}")
print(f"  min-rule deviation / LP error: min {np.min((2*pred)/np.maximum(errs,1e-12)):.8f}, "
      f"max {np.max((2*pred)/np.maximum(errs,1e-12)):.8f}")
OUT["linf_identity"] = float(np.max(np.abs(errs - pred)))

print()
print("=" * 80)
print("(G2) decreasing target: bias >= max_delta [ b_{n(delta)} - delta ]")
print("=" * 80)
print(f"{'K':>5s} {'drop':>6s} {'bias':>8s} {'bound':>8s} {'delta*':>7s} "
      f"{'n*':>4s} {'ok':>4s}")
worst = np.inf
for K in (100, 1000):
    for drop in (0.0, 2.0, 4.0, 8.0, 16.0, 64.0, 512.0):
        t = np.linspace(drop, 0.0, K)
        reps = int(max(20000, 2000000 // K))
        bias = float(t[-1] - np.mean(
            np.min(t[None, :] + rng.standard_normal((reps, K)), axis=1)))
        best, bd, bn = -np.inf, 0.0, K
        for d in np.linspace(0.0, drop + 1.0, 600):
            n = int(np.sum(t - t[-1] <= d))
            if n < 1:
                continue
            lb = b_exact(n) - d
            if lb > best:
                best, bd, bn = lb, d, n
        worst = min(worst, bias - best)
        print(f"{K:5d} {drop:6.1f} {bias:8.4f} {best:8.4f} {bd:7.3f} {bn:4d} "
              f"{str(bias >= best - 1e-3):>4s}")
print(f"  worst margin (bias - bound) = {worst:.4f}  (must be >= -mc error)")
OUT["lower_bound_margin"] = float(worst)

print()
print("=" * 80)
print("(G3) correlated estimates, AR(1) noise: price vs effective block length")
print("=" * 80)
print(f"{'K':>5s} {'rho':>5s} {'bias':>8s} {'b_K':>8s} {'Ke_=K(1-r)/(1+r)':>17s} "
      f"{'b_Ke':>8s}")
G = []
for K in (200, 1000):
    base = b_exact(K)
    for rho in (0.0, 0.5, 0.9, 0.99):
        reps = int(max(20000, 2000000 // K))
        e = np.empty((reps, K))
        e[:, 0] = rng.standard_normal(reps)
        s = sqrt(1 - rho ** 2)
        for j in range(1, K):
            e[:, j] = rho * e[:, j - 1] + s * rng.standard_normal(reps)
        bias = float(-np.mean(np.min(e, axis=1)))
        ke = max(1.0, K * (1 - rho) / (1 + rho))
        pred = b_exact(max(2, int(round(ke))))
        G.append(dict(K=K, rho=rho, bias=bias, Ke=ke, pred=pred))
        print(f"{K:5d} {rho:5.2f} {bias:8.4f} {base:8.4f} {ke:17.1f} {pred:8.4f}")
OUT["ar1"] = G

print()
print("=" * 80)
print("(G4) MSE under contamination: min rule vs tempered rules (K=50, Delta=8)")
print("=" * 80)
K, Delta, reps = 50, 8.0, 20000
bK = b_exact(K)
print(f"  b_K = {bK:.4f}")
print(f"  {'eps':>5s} {'rule':>16s} {'E[theta_K]':>11s} {'bias':>7s} {'sd':>7s} {'rmse':>7s}")
for eps in (0.0, 0.005, 0.02, 0.05):
    th = rng.standard_normal((reps, K))
    th = th - Delta * (rng.random((reps, K)) < eps)
    rm = np.minimum.accumulate(th, axis=1)
    for name, y in (("min (delta=0)", rm),
                    ("min+b_K/2", rm + bK / 2),
                    ("min+b_K", rm + bK),
                    ("min+2b_K", rm + 2 * bK)):
        e = y[:, -1]
        print(f"  {eps:5.3f} {name:>16s} {e.mean():11.4f} {-e.mean():7.4f} "
              f"{e.std():7.4f} {np.sqrt(np.mean(e**2)):7.4f}")
    print()

print("=" * 80)
print("(G5) causality attack: is any causal feasible schedule above the min rule?")
print("=" * 80)
hits = 0
for _ in range(20000):
    K = int(rng.integers(2, 20))
    t = rng.standard_normal(K)
    base = np.minimum.accumulate(t)
    y = base.copy()
    j = int(rng.integers(K))
    y[j] += float(rng.uniform(1e-9, 1.0))
    ok = True
    for m in range(j + 1, K):                 # keep causality + feasibility later on
        y[m] = min(y[m], y[m - 1])
    ok = np.all(y <= t + 1e-12) and np.all(np.diff(y) <= 1e-12)
    if ok and np.any(y > base + 1e-12):
        hits += 1
print(f"  causal feasible schedules strictly above the min rule: {hits}/20000")

os.makedirs("results", exist_ok=True)
with open(os.path.join("results", "closing.json"), "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=2, default=float)
print()
print("written: results/closing.json")
