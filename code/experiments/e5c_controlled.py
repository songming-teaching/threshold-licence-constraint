"""E5c: resolve the E5 / E5b inconsistency.

E5 (anneal with force_mask, full K=200 iterations) gave regime-B ordering
    no-truncation 0.0495  <  published 0.0895  <  contamination-aware 0.0948
E5b (settle 8 sweeps from the published solution) gave
    published 0.0895  ~=  contamination-aware 0.0894  <  count-matched 0.1007

Same strategy, different answer. The difference is the procedure: E5 runs the whole
annealing with a forced mask (so the threshold path itself diverges), E5b only
re-settles a few sweeps.

This experiment removes the confound: fixed threshold for every arm, common
initialisation, long settle, only the keep-mask differs.
"""
import os
import sys

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pbcd import madn, pbcd_step   # noqa: E402

EXP = os.path.dirname(HERE)
DATA = os.path.join(EXP, "data")
os.makedirs(DATA, exist_ok=True)

SEED = 20261102
N, RANK, P_OBS, K = 100, 3, 0.6, 200
REPS = 20
SETTLE = 60


def make(rng, n, rank, p, sigma, regime, frac, mag):
    U = rng.standard_normal((n, rank)) / np.sqrt(rank)
    V = rng.standard_normal((n, rank)) / np.sqrt(rank)
    X = U @ V.T
    mask = rng.random((n, n)) < p
    Y = np.where(mask, X + sigma * rng.standard_normal((n, n)), 0.0)
    bad = np.zeros_like(mask)
    if regime == "A":
        bad = mask & (rng.random((n, n)) < frac)
    elif regime == "B":
        bad = mask & (rng.random(n) < frac)[:, None]
    Y = Y + np.where(bad, mag * sigma * rng.uniform(0.5, 1.5, (n, n)), 0.0)
    return X, Y, mask, bad


def init_winsor(Y, mask, rank, z=3.0):
    v = Y[mask]
    s = 1.4826 * np.median(np.abs(v - np.median(v)))
    Z = np.where(mask, np.clip(Y, -z * s, z * s), 0.0)
    U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    return U[:, :rank] * np.sqrt(sv[:rank]), Vt[:rank].T * np.sqrt(sv[:rank])


def settle_from(Y, keep, U0, V0, theta, lam, steps):
    U, V = U0.copy(), V0.copy()
    for _ in range(steps):
        U, V = pbcd_step(Y, keep, U, V, theta, lam, 1)
    return U, V


def one(regime, frac, rep, mag=6.0, sigma=0.05):
    rng = np.random.default_rng(SEED + 7919 * rep + int(frac * 1000)
                                + {"A": 0, "B": 1}[regime])
    X, Y, mask, bad = make(rng, N, RANK, P_OBS, sigma, regime, frac, mag)
    lam = 1e-3 * float(np.trace(Y.T @ Y) / mask.sum())
    nX = np.linalg.norm(X)

    # reference threshold path, once
    U, V = init_winsor(Y, mask, RANK)
    cur = 1e18
    for _ in range(5):
        U, V = pbcd_step(Y, mask, U, V, cur, lam, 1)
    that = []
    for _ in range(K):
        U, V = pbcd_step(Y, mask, U, V, cur, lam, 1)
        R = (Y - U @ V.T)[mask]
        cur = min(cur, madn(R)); that.append(cur)
    theta = float(that[-1])

    # common start for every arm
    U0, V0 = init_winsor(Y, mask, RANK)
    for _ in range(5):
        U0, V0 = pbcd_step(Y, mask, U0, V0, theta, lam, 1)

    arms = {}
    # (1) keep everything, including corrupted
    arms["keep-all"] = settle_from(Y, mask, U0, V0, theta, lam, SETTLE)
    # (2) the published rule's kept set at this threshold
    R = Y - U0 @ V0.T
    keep_pub = mask & (np.abs(R) <= theta)
    arms["threshold-kept"] = settle_from(Y, keep_pub, U0, V0, theta, lam, SETTLE)
    # (3) count-matched random
    keep_rand = np.zeros_like(mask)
    nk = keep_pub.sum(axis=1)
    for i in range(N):
        cand = np.flatnonzero(mask[i]); k = int(min(nk[i], cand.size))
        if k:
            keep_rand[i, rng.choice(cand, size=k, replace=False)] = True
    arms["count-matched"] = settle_from(Y, keep_rand, U0, V0, theta, lam, SETTLE)
    # (4) contamination-aware deletion
    keep_ca = mask & ~bad
    arms["contam-aware"] = settle_from(Y, keep_ca, U0, V0, theta, lam, SETTLE)

    out = []
    for name, (Uu, Vv) in arms.items():
        out.append(dict(regime=regime, frac=frac, rep=rep, arm=name,
                        relerr=float(np.linalg.norm(X - Uu @ Vv.T) / nX),
                        theta=theta))
    return out


if __name__ == "__main__":
    print(f"E5c controlled comparison  n={N} rank={RANK} p={P_OBS} "
          f"settle={SETTLE} reps={REPS}")
    print("       fixed threshold, common init, only the keep-mask differs")
    jobs = [(rg, f, r) for rg in ("A", "B") for f in (0.05, 0.10, 0.20)
            for r in range(REPS)]
    res = Parallel(n_jobs=8)(delayed(one)(rg, f, r) for rg, f, r in jobs)
    rows = [x for sub in res for x in sub]

    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(DATA, "e5c_controlled.csv")
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write(f"# E5c controlled | seed={SEED} | n={N} rank={RANK} p={P_OBS} "
                 f"settle={SETTLE} reps={REPS}\n")
        fh.write("# fixed threshold, common initialisation, only the keep-mask "
                 "differs -> removes the E5/E5b procedural confound\n")
    df.to_csv(csv, mode="a", index=False)

    print()
    print("=" * 88)
    print(f"{'regime':>7s} {'frac':>5s} " +
          " ".join(f"{a:>16s}" for a in
                   ("keep-all", "threshold-kept", "count-matched", "contam-aware")))
    for rg in ("A", "B"):
        for f in (0.05, 0.10, 0.20):
            cells = []
            for a in ("keep-all", "threshold-kept", "count-matched", "contam-aware"):
                q = df[(df.regime == rg) & np.isclose(df.frac, f) & (df.arm == a)]
                cells.append(f"{q.relerr.mean():10.4f}±{q.relerr.std(ddof=1)/np.sqrt(len(q)):.4f}")
            print(f"{rg:>7s} {f:5.2f} " + " ".join(cells))
        print()

    print("=" * 88)
    print("paired tests against threshold-kept (negative = the other arm is better)")
    print("=" * 88)
    for rg in ("A", "B"):
        for f in (0.05, 0.10, 0.20):
            s = df[(df.regime == rg) & np.isclose(df.frac, f)]
            base = s[s.arm == "threshold-kept"].set_index("rep").relerr
            line = f"  {rg} frac={f:4.2f}: "
            for a in ("keep-all", "count-matched", "contam-aware"):
                q = s[s.arm == a].set_index("rep").relerr
                d = q - base
                t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) else 0
                line += f"{a} {d.mean():+.4f}(t{t:+.1f})  "
            print(line)
