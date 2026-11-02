"""Log the threshold path per iteration, so the effective-history question is a rerun.

Writes, for each (eps, rep): the full thetahat_1..K path, the applied schedule, the
unconstrained final estimate, and three candidate effective lengths:
  n_total   : the iteration count K
  n_trans   : first index at which the path comes within 1% of its terminal level,
              i.e. the end of the transient
  n_plateau : number of iterations before the value stops changing at all
              (last index with a distinct value)

Then compares b computed under each against the bias the solver actually records.
"""
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from math import log, sqrt
from scipy.integrate import quad
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pbcd import madn, pbcd_step   # noqa: E402

EXP = os.path.dirname(HERE)
DATA = os.path.join(EXP, "data")
os.makedirs(DATA, exist_ok=True)

SEED = 20261102
N, RANK, P_OBS, K, INNER = 100, 3, 0.6, 200, 1
REPS = 20
EPS_LIST = (0.0, 0.02, 0.05, 0.10)


def b_exact(k):
    if k < 2:
        return 0.0
    f = lambda t: 1.0 - norm.cdf(t) ** k - norm.sf(t) ** k
    return float(quad(f, 0.0, sqrt(2 * log(k)) + 14.0, limit=600)[0])


def make(rng, n, rank, p, sigma, eps, mag):
    U = rng.standard_normal((n, rank)) / np.sqrt(rank)
    V = rng.standard_normal((n, rank)) / np.sqrt(rank)
    X = U @ V.T
    mask = rng.random((n, n)) < p
    Y = np.where(mask, X + sigma * rng.standard_normal((n, n)), 0.0)
    bad = mask & (rng.random((n, n)) < eps)
    Y[bad] += mag * sigma * rng.uniform(0.5, 1.5, bad.sum())
    return X, Y, mask, bad


def init_winsor(Y, mask, rank, z=3.0):
    v = Y[mask]
    s = 1.4826 * np.median(np.abs(v - np.median(v)))
    Z = np.where(mask, np.clip(Y, -z * s, z * s), 0.0)
    U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    return U[:, :rank] * np.sqrt(sv[:rank]), Vt[:rank].T * np.sqrt(sv[:rank])


def one(eps, rep, mag=6.0, sigma=0.05):
    rng = np.random.default_rng(SEED + 1000 * rep + int(eps * 1000))
    X, Y, mask, bad = make(rng, N, RANK, P_OBS, sigma, eps, mag)
    lam = 1e-3 * float(np.trace(Y.T @ Y) / mask.sum())

    U, V = init_winsor(Y, mask, RANK)
    cur = 1e18
    for _ in range(5):
        U, V = pbcd_step(Y, mask, U, V, cur, lam, INNER)

    that, theta = [], []
    for _ in range(K):
        for _ in range(INNER):
            U, V = pbcd_step(Y, mask, U, V, cur, lam)
        R = (Y - U @ V.T)[mask]
        h = madn(R)
        cur = min(h, cur)
        that.append(h)
        theta.append(cur)
    that = np.asarray(that)
    theta = np.asarray(theta)

    term = float(np.median(that[-20:]))
    # n_trans: first index within 1% of the terminal level
    hits = np.where(np.abs(that - term) <= 0.01 * abs(term))[0]
    n_trans = int(hits[0] + 1) if hits.size else K
    # n_plateau: last index where the rounded value differs from the next
    d = np.where(np.abs(np.diff(that)) > 1e-12)[0]
    n_plateau = int(d[-1] + 2) if d.size else 1

    relerr = float(np.linalg.norm(X - U @ V.T) / np.linalg.norm(X))
    return dict(eps=eps, rep=rep, theta_path=theta.tolist(),
                that_path=that.tolist(), terminal=term,
                n_total=K, n_trans=n_trans, n_plateau=n_plateau,
                relerr=relerr, madn_final=float(that[-1]))


if __name__ == "__main__":
    print(f"logging threshold paths: n={N} rank={RANK} K={K} reps={REPS}")
    jobs = [(e, r) for e in EPS_LIST for r in range(REPS)]
    res = Parallel(n_jobs=8)(delayed(one)(e, r) for e, r in jobs)

    out = os.path.join(DATA, "e13_threshold_paths.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(seed=SEED, n=N, rank=RANK, p=P_OBS, K=K, inner=INNER,
                       reps=REPS, eps_list=list(EPS_LIST),
                       b_K={str(k): b_exact(k) for k in
                            (1, 2, 3, 5, 10, 20, 50, 200)},
                       runs=res), fh)
    print("written:", out)

    print()
    print("=" * 92)
    print("Effective history: three candidate definitions, per contamination level")
    print("=" * 92)
    print(f"  {'eps':>5s} {'n_total':>8s} {'n_trans':>8s} {'n_plateau':>10s} "
          f"{'b(n_total)':>11s} {'b(n_trans)':>11s} {'observed gap':>13s}")
    for eps in EPS_LIST:
        runs = [r for r in res if r["eps"] == eps]
        nt = np.mean([r["n_total"] for r in runs])
        ntr = np.mean([r["n_trans"] for r in runs])
        np_ = np.mean([r["n_plateau"] for r in runs])
        gaps = [r["terminal"] - float(np.mean(r["theta_path"])) for r in runs]
        print(f"  {eps:5.2f} {nt:8.1f} {ntr:8.2f} {np_:10.2f} "
              f"{b_exact(int(nt)):11.4f} {b_exact(int(round(ntr))):11.4f} "
              f"{np.mean(gaps):13.5f}")
    print()
    print("  observed gap = mean(terminal estimate) - mean(schedule), in solver units.")
    print("  b is in units of the estimator spread, so the two are not directly")
    print("  comparable; the point of the table is the ORDER of n_trans vs n_total.")
