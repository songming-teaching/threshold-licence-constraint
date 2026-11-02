"""Consolidated verification for the paper's core. One ensemble, one seed.

Produces every number quoted in the main body, with Monte-Carlo standard errors.
Writes results/core.json.

Sections
  A  characterisation: pointwise maximality of the truncated running minimum
  B  the frontier: bias + excess >= b_K, equality iff shifted running minimum
  C  the price: Gaussian / heavy tail / multi-output
  D  breakdown: one downward outlier poisons the whole tail
"""
import json
import os
from math import gamma, log, pi, sqrt

import numpy as np
from scipy.optimize import linprog

SEED = 20261101
rng = np.random.default_rng(SEED)
OUT = {"seed": SEED}

# ------------------------------------------------------------------ helpers


def run_min(a, delta=0.0):
    return np.minimum.accumulate(a + delta, axis=-1)


def iso_l2(th):
    vals, wts = [], []
    for t in th:
        vals.append(float(t)); wts.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2)
    return np.array([v for v, w in zip(vals, wts) for _ in range(w)])


def iso_linf(th):
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


def bias_excess(y, th):
    """(bias, excess) for a batch. y, th shape (reps, k)."""
    b = float(-np.mean(y[:, -1]))
    x = float(np.mean(np.max(np.maximum(y - th, 0.0), axis=1)))
    return b, x


def mcse(x):
    return float(np.std(x, ddof=1) / sqrt(len(x)))


# -------------------------------------------------- A: pointwise maximality
print("=" * 78)
print("A. Characterisation: trunced running min is the pointwise maximum")
print("=" * 78)
worst = 0.0
for _ in range(400):
    k = int(rng.integers(3, 40))
    th = rng.standard_normal(k)
    d = float(rng.uniform(0.0, 1.0))
    y = run_min(th, d)
    # feasible: non-increasing, below th+d
    assert np.all(np.diff(y) <= 1e-12)
    assert np.all(y <= th + d + 1e-12)
    # maximal: any feasible z with z_j > y_j somewhere is impossible
    for _ in range(20):
        j = int(rng.integers(k))
        z = y.copy(); z[j] += 1e-6
        z = np.minimum.accumulate(z)          # restore monotonicity
        if np.all(z <= th + d + 1e-12):
            worst = max(worst, float(np.max(z - y)))
print(f"  400 sequences x 20 one-coordinate pushes: max escapable increment "
      f"= {worst:.2e}")
OUT["maximality_escape"] = worst

# counterexample: running min is not any L_p isotonic projection
print()
print("  counterexample (not a projection): theta_hat = (0, 3, 1, 2)")
th = np.array([0.0, 3.0, 1.0, 2.0])
print(f"    running min        = {run_min(th)}")
print(f"    L2 isotonic        = {iso_l2(th)}")
print(f"    Linf isotonic      = {np.round(iso_linf(th), 4)}")
OUT["counterexample"] = dict(th=th.tolist(), min=run_min(th).tolist(),
                             l2=iso_l2(th).tolist(),
                             linf=iso_linf(th).tolist())

# ratio of L-inf deviations, over many paths
lr = []
for _ in range(2000):
    k = int(rng.integers(4, 30))
    t = rng.standard_normal(k)
    a = np.max(np.abs(run_min(t) - t))
    b = np.max(np.abs(iso_linf(t) - t))
    if b > 1e-9:
        lr.append(a / b)
print(f"    L-inf deviation ratio min/iso over 2000 paths: "
      f"median {np.median(lr):.4f}, 10%-90% [{np.percentile(lr,10):.3f}, "
      f"{np.percentile(lr,90):.3f}]")
OUT["linf_ratio"] = dict(median=float(np.median(lr)),
                         q10=float(np.percentile(lr, 10)),
                         q90=float(np.percentile(lr, 90)))

# ------------------------------------------------------------- B: frontier
print()
print("=" * 78)
print("B. Frontier: bias + excess >= b_K, equality iff shifted running min")
print("=" * 78)
print(f"{'K':>5s} {'reps':>7s} {'b_K':>8s} {'se':>7s} | "
      f"{'schedule':>14s} {'bias':>8s} {'excess':>8s} {'sum':>8s} {'minus b_K':>10s}")
B = []
for K in (10, 40, 150):
    reps = max(2000, 60000 // K)
    th = rng.standard_normal((reps, K))
    bK = -np.mean(np.min(th, axis=1))
    seK = mcse(-np.min(th, axis=1))
    rows = []
    cand = {"running min": run_min(th, 0.0),
            "min + 0.5": run_min(th, 0.5),
            "min + b_K": run_min(th, bK),
            "Linf iso": None, "L2 iso": None}
    iso_l = np.array([iso_linf(t) for t in th])
    iso_2 = np.array([iso_l2(t) for t in th])
    cand["Linf iso"] = iso_l
    cand["L2 iso"] = iso_2
    for name, y in cand.items():
        b, x = bias_excess(y, th)
        rows.append(dict(schedule=name, bias=b, excess=x, sum=b + x))
        print(f"{K:5d} {reps:7d} {bK:8.4f} {seK:7.4f} | {name:>14s} "
              f"{b:8.4f} {x:8.4f} {b+x:8.4f} {b+x-bK:+10.4f}")
    B.append(dict(K=K, reps=reps, bK=bK, seK=seK, rows=rows))
    print()
OUT["frontier"] = B

print("  exchange rate, sample by sample (K=40, same ensemble):")
K = 40
th = rng.standard_normal((20000, K))
bK = -np.mean(np.min(th, axis=1))
print(f"    {'delta':>7s} {'bias':>9s} {'bias drop':>10s} {'excess':>8s} {'sum':>8s}")
E = []
for d in (0.0, 0.1, 0.5, 1.0, 2.0, 3.0):
    y = run_min(th, d)
    b = -np.mean(y[:, -1])
    x = float(np.mean(np.max(np.maximum(y - th, 0.0), axis=1)))
    E.append(dict(delta=d, bias=float(b), excess=x, sum=float(b) + x))
    print(f"    {d:7.2f} {b:9.4f} {bK-b:10.4f} {x:8.6f} {b+x:8.4f}")
OUT["exchange"] = E

# ---------------------------------------------------------------- C: price
print()
print("=" * 78)
print("C. Price of monotonicity")
print("=" * 78)
print(f"{'K':>6s} {'reps':>8s} {'b_K':>8s} {'sqrt2lnK':>9s} {'1-term':>8s} "
      f"{'Gumbel':>8s}")
G = []
for K in (10, 32, 100, 316, 1000, 3162, 10000):
    reps = int(max(4000, 2000000 // K))
    b = -np.mean(np.min(rng.standard_normal((reps, K)), axis=1))
    a = sqrt(2.0 * log(K))
    one = a - log(log(K)) / (2.0 * a)
    gum = a - (log(log(K)) + log(4.0 * pi)) / (2.0 * a)
    G.append(dict(K=K, reps=reps, bias=float(b), first=a, one=one, gumbel=gum))
    print(f"{K:6d} {reps:8d} {b:8.4f} {a:9.4f} {one:8.4f} {gum:8.4f}")
OUT["gauss"] = G

e = rng.standard_normal((4000000, 2))
b2 = -np.mean(np.min(e, axis=1))
print(f"  floor: b_2 = {b2:.4f} (se {mcse(-np.min(e,axis=1)):.4f}), "
      f"1/sqrt(pi) = {1/sqrt(pi):.4f}")
OUT["floor_b2"] = float(b2)


def t_unit(nu, shape):
    z = rng.standard_normal(shape)
    c = rng.chisquare(nu, shape)
    return z / np.sqrt(c / nu) * sqrt((nu - 2.0) / nu)


print()
print("  heavy tails (unit-variance t_nu, tail constant folded in):")
print(f"  {'nu':>3s} {'K':>6s} {'b_K':>10s} {'(KC)^(1/nu)G(1-1/nu)':>22s} {'ratio':>7s}")
H = []
for nu in (3, 4, 6):
    C = (gamma((nu + 1) / 2.0) * nu ** (nu / 2.0 - 1.0)
         / (sqrt(pi) * gamma(nu / 2.0))) * ((nu - 2.0) / nu) ** (nu / 2.0)
    for K in (100, 1000, 10000):
        reps = int(max(4000, 2000000 // K))
        b = -np.mean(np.min(t_unit(nu, (reps, K)), axis=1))
        pred = (K * C) ** (1.0 / nu) * gamma(1.0 - 1.0 / nu)
        H.append(dict(nu=nu, K=K, bias=float(b), pred=pred, ratio=float(b / pred)))
        print(f"  {nu:3d} {K:6d} {b:10.4f} {pred:22.4f} {b/pred:7.3f}")
OUT["heavy"] = H

print()
print("  multi-output, d coordinatewise thresholds:")
print(f"  {'d':>4s} {'K':>6s} {'b':>8s} {'sqrt(2ln dK)':>13s} {'sqrt(2lnK)':>11s}")
M = []
for d in (1, 10, 100):
    for K in (100, 1000):
        reps = int(max(2000, 1000000 // K))
        e = rng.standard_normal((reps, d, K))
        b = float(-np.mean(np.min(e.reshape(reps, -1), axis=1)))
        M.append(dict(d=d, K=K, bias=b, pred_dK=sqrt(2.0 * log(d * K)),
                      pred_K=sqrt(2.0 * log(K))))
        print(f"  {d:4d} {K:6d} {b:8.4f} {sqrt(2.0*log(d*K)):13.4f} "
              f"{sqrt(2.0*log(K)):11.4f}")
OUT["multi"] = M

# ------------------------------------------------------------ D: breakdown
print()
print("=" * 78)
print("D. Breakdown: the running min has breakdown point 1/K")
print("=" * 78)
K, Delta, reps = 200, 8.0, 20000
print("  target flat at 0; each step shifted by -Delta with prob eps")
print(f"  {'eps':>6s} {'P(outlier)':>11s} {'min rule':>10s} {'min|poisoned':>14s} "
      f"{'capped at -c':>14s} {'c = b_K':>9s}")
D = []
_a = sqrt(2.0 * log(K))
bK200 = float(_a - (log(log(K)) + log(4.0 * pi)) / (2.0 * _a))
for eps in (0.0, 0.002, 0.01, 0.05):
    th = rng.standard_normal((reps, K))
    bad = rng.random((reps, K)) < eps
    th = th - Delta * bad
    rm = np.minimum.accumulate(th, axis=1)
    hit = np.any(bad, axis=1)
    yc = np.maximum(rm, -bK200)
    D.append(dict(eps=eps, p_out=float(1 - (1 - eps) ** K),
                  min_mean=float(rm[:, -1].mean()),
                  min_poisoned=float(rm[hit, -1].mean()) if hit.any() else None,
                  capped_mean=float(yc[:, -1].mean())))
    print(f"  {eps:6.3f} {1-(1-eps)**K:11.3f} {rm[:,-1].mean():10.4f} "
          f"{(rm[hit,-1].mean() if hit.any() else float('nan')):14.4f} "
          f"{yc[:,-1].mean():14.4f} {-bK200:9.4f}")
OUT["breakdown"] = dict(K=K, Delta=Delta, cap=-bK200, rows=D)

os.makedirs("results", exist_ok=True)
with open(os.path.join("results", "core.json"), "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=2)
print()
print("written: results/core.json")
