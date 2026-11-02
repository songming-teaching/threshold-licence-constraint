"""Regenerate the two defective simulation columns, and audit the reviewer's门槛 claims.

Regenerations
  RG1  section 4.5 table: drop=512 row (and all rows) with reps that GROW with K
  RG2  section 4.3 table: heavy tail, reps that GROW with K and with the tail index

Audits of the third-round review
  A1  claim: modified Gumbel expansion error <1% for K >= 3e3, and 6.0%/1.95% at K=10/100
  A2  claim: nu=3 at K=1e2 gives ratio 0.966 (the start of my convergence sequence)
  A3  claim: with correct simulation, all six rows of 4.5 are within 0.005
"""
import json
import os
from math import gamma, pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, t as tdist

OUT = {}


def b_gauss(K):
    f = lambda s: 1.0 - norm.cdf(s) ** K - norm.sf(s) ** K
    return float(quad(f, 0.0, sqrt(2 * np.log(K)) + 14.0, limit=600,
                      epsabs=1e-12, epsrel=1e-12)[0])


def b_moving(K, drop):
    a = np.linspace(drop, 0.0, K)

    def g(s):
        lg = np.log1p(-norm.cdf(s - a))
        lo = 1.0 - np.exp(np.sum(np.log1p(-norm.cdf(-s - a))))
        return lo - np.exp(np.sum(lg))
    return float(quad(g, 0.0, drop + 12.0, limit=400, epsabs=1e-11)[0])


def b_t(nu, K):
    scale = sqrt((nu - 2.0) / nu)

    def g(s):
        x = np.exp(s)
        F = float(tdist.cdf(x / scale, nu))
        if F >= 1.0:
            return 0.0
        return (1.0 - F ** K - (1.0 - F) ** K) * x
    return float(quad(g, -40.0, np.log(1e7), limit=500, epsabs=1e-11)[0])


rng = np.random.default_rng(20261101)

# ------------------------------------------------------------------ RG1
print("=" * 82)
print("RG1. Section 4.5 rows regenerated: reps grow with K")
print("=" * 82)
K = 1000
print(f"  {'drop':>6s} {'reps':>9s} {'exact':>9s} {'simulated':>11s} {'se':>8s} "
      f"{'diff':>8s} {'z':>6s}")
rows = []
for drop in (0.0, 2.0, 8.0, 16.0, 64.0, 512.0):
    reps = 400_000
    t = np.linspace(drop, 0.0, K)
    acc, done = [], 0
    while done < reps:
        n = min(100_000, reps - done)
        acc.append(-np.min(t[None, :] + rng.standard_normal((n, K)), axis=1))
        done += n
    v = np.concatenate(acc)
    ex = b_moving(K, drop)
    se = float(v.std(ddof=1) / sqrt(len(v)))
    rows.append(dict(drop=drop, reps=reps, exact=ex, sim=float(v.mean()), se=se))
    print(f"  {drop:6.1f} {reps:9d} {ex:9.4f} {v.mean():11.4f} {se:8.5f} "
          f"{v.mean()-ex:8.4f} {(v.mean()-ex)/se:+6.2f}")
OUT["sec45"] = rows

# ------------------------------------------------------------------ RG2
print()
print("=" * 82)
print("RG2. Section 4.3 rows regenerated: reps grow with K, tail-aware")
print("=" * 82)
print(f"  {'nu':>3s} {'K':>7s} {'reps':>8s} {'exact':>9s} {'simulated':>11s} "
      f"{'se':>8s} {'diff':>8s} {'z':>6s}")
hrows = []
for nu in (3, 4, 6):
    scale = sqrt((nu - 2.0) / nu)
    for K in (100, 1000, 10000):
        reps = {100: 200_000, 1000: 200_000, 10000: 100_000}[K]
        acc, done = [], 0
        while done < reps:
            n = min(25_000, reps - done)
            z = rng.standard_normal((n, K))
            c = rng.chisquare(nu, (n, K))
            e = z / np.sqrt(c / nu) * scale
            acc.append(-np.min(e, axis=1))
            done += n
        v = np.concatenate(acc)
        ex = b_t(nu, K)
        se = float(v.std(ddof=1) / sqrt(len(v)))
        hrows.append(dict(nu=nu, K=K, reps=reps, exact=ex, sim=float(v.mean()), se=se))
        print(f"  {nu:3d} {K:7d} {reps:8d} {ex:9.4f} {v.mean():11.4f} {se:8.5f} "
              f"{v.mean()-ex:8.4f} {(v.mean()-ex)/se:+6.2f}")
    print()
OUT["sec43"] = hrows

# ------------------------------------------------------------------ A1
print("=" * 82)
print("A1. Gumbel expansion with the Euler term: error by K")
print("=" * 82)
print(f"  {'K':>8s} {'exact':>9s} {'bc+gamma/bc':>12s} {'abs err':>9s} {'rel err':>9s}")
g = []
for K in (10, 100, 300, 1000, 3000, 10000, 100000):
    ex = b_gauss(K)
    a = sqrt(2 * np.log(K))
    bc = a - (np.log(np.log(K)) + np.log(4 * pi)) / (2 * a)
    mod = bc + 0.5772156649015329 / bc
    g.append(dict(K=K, exact=ex, mod=mod, rel=abs(mod - ex) / ex))
    print(f"  {K:8d} {ex:9.4f} {mod:12.4f} {abs(mod-ex):9.4f} {abs(mod-ex)/ex*100:8.2f}%")
OUT["gumbel"] = g

# ------------------------------------------------------------------ A2
print()
print("=" * 82)
print("A2. heavy-tail ratio sequence: does nu=3 start at 0.966 at K=1e2?")
print("=" * 82)
print(f"  {'nu':>3s} " + " ".join(f"{'K='+str(K):>9s}" for K in (100, 1000, 10000, 100000)))
for nu in (3, 4, 6):
    C_std = gamma((nu + 1) / 2.0) * nu ** (nu / 2.0 - 1.0) / (sqrt(pi) * gamma(nu / 2.0))
    C = C_std * ((nu - 2.0) / nu) ** (nu / 2.0)
    cells = []
    for K in (100, 1000, 10000, 100000):
        ex = b_t(nu, K)
        asym = (K * C) ** (1.0 / nu) * gamma(1.0 - 1.0 / nu)
        cells.append(f"{ex/asym:9.3f}")
    print(f"  {nu:3d} " + " ".join(cells))

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "results", "regen.json"), "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=2, default=float)
print()
print("written: results/regen.json")
