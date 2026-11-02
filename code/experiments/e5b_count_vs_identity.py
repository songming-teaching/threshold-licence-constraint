"""E5b: separate "how many observations are kept" from "which ones are kept".

E5 showed the ordering of three deletion strategies reverses between independent and
whole-row contamination, but it varied both the count and the identity at once.
This experiment fixes the count.

In regime B, for each row:
  published      : keep entries selected by the MADN threshold
  count-matched  : keep the SAME NUMBER per row, chosen uniformly at random
                   (ignores which entries are actually good)
  contamination-aware : keep every clean entry, drop every corrupted one

If count-matched is close to published, the count dominates.
If count-matched is close to contamination-aware, identity dominates.

Also reports the post-truncation coverage (retained observations per row).
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
N, RANK, P_OBS, K, INNER = 100, 3, 0.6, 200, 1
REPS = 20


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
        rows = rng.random(n) < frac
        bad = mask & rows[:, None]
    Y = Y + np.where(bad, mag * sigma * rng.uniform(0.5, 1.5, (n, n)), 0.0)
    return X, Y, mask, bad


def init_winsor(Y, mask, rank, z=3.0):
    v = Y[mask]
    s = 1.4826 * np.median(np.abs(v - np.median(v)))
    Z = np.where(mask, np.clip(Y, -z * s, z * s), 0.0)
    U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    return U[:, :rank] * np.sqrt(sv[:rank]), Vt[:rank].T * np.sqrt(sv[:rank])


def anneal(Y, mask, rank, lam):
    U, V = init_winsor(Y, mask, rank)
    cur = 1e18
    for _ in range(5):
        U, V = pbcd_step(Y, mask, U, V, cur, lam, INNER)
    that = []
    for _ in range(K):
        for _ in range(INNER):
            U, V = pbcd_step(Y, mask, U, V, cur, lam)
        R = (Y - U @ V.T)[mask]
        h = madn(R)
        cur = min(h, cur)
        that.append(h)
    return np.asarray(that), U, V


def settle(Y, mask, rank, lam, U, V, theta, keep):
    """Run a few proximal sweeps restricted to a given keep-mask."""
    U, V = U.copy(), V.copy()
    for _ in range(8):
        U, V = pbcd_step(Y, mask, U, V, theta, lam, INNER) if keep is None else \
              pbcd_step(Y, keep, U, V, theta, lam, INNER)
    return U, V


def one(regime, frac, rep, mag=6.0, sigma=0.05):
    rng = np.random.default_rng(SEED + 7919 * rep + int(frac * 1000)
                                + {"A": 0, "B": 1}[regime])
    X, Y, mask, bad = make(rng, N, RANK, P_OBS, sigma, regime, frac, mag)
    lam = 1e-3 * float(np.trace(Y.T @ Y) / mask.sum())
    nX = np.linalg.norm(X)

    that, U, V = anneal(Y, mask, RANK, lam)
    theta = float(np.minimum.accumulate(that)[-1])
    R = Y - U @ V.T
    keep_pub = mask & (np.abs(R) <= theta)          # the published rule's kept set
    n_keep = keep_pub.sum(axis=1)                    # per-row count to match

    out = []
    # ---- published
    Up, Vp = settle(Y, mask, RANK, lam, U, V, theta, keep_pub)
    out.append(dict(regime=regime, frac=frac, rep=rep, arm="published",
                    relerr=float(np.linalg.norm(X - Up @ Vp.T) / nX),
                    coverage=float(np.mean(keep_pub.sum(axis=1))),
                    cov_bad_rows=float(np.mean(keep_pub[bad.any(axis=1)].sum(axis=1)))
                    if bad.any() else np.nan))

    # ---- count-matched random selection, same per-row count
    keep_rand = np.zeros_like(mask)
    for i in range(N):
        cand = np.flatnonzero(mask[i])
        k = int(min(n_keep[i], cand.size))
        if k > 0:
            keep_rand[i, rng.choice(cand, size=k, replace=False)] = True
    Ur, Vr = settle(Y, mask, RANK, lam, U, V, theta, keep_rand)
    out.append(dict(regime=regime, frac=frac, rep=rep, arm="count-matched",
                    relerr=float(np.linalg.norm(X - Ur @ Vr.T) / nX),
                    coverage=float(np.mean(keep_rand.sum(axis=1))),
                    cov_bad_rows=float(np.mean(keep_rand[bad.any(axis=1)].sum(axis=1)))
                    if bad.any() else np.nan))

    # ---- contamination-aware deletion
    keep_ca = mask & ~bad
    Uc, Vc = settle(Y, mask, RANK, lam, U, V, theta, keep_ca)
    out.append(dict(regime=regime, frac=frac, rep=rep, arm="contam-aware",
                    relerr=float(np.linalg.norm(X - Uc @ Vc.T) / nX),
                    coverage=float(np.mean(keep_ca.sum(axis=1))),
                    cov_bad_rows=float(np.mean(keep_ca[bad.any(axis=1)].sum(axis=1)))
                    if bad.any() else np.nan))
    return out


if __name__ == "__main__":
    print(f"E5b count vs identity  n={N} rank={RANK} p={P_OBS} reps={REPS}")
    jobs = [(rg, f, r) for rg in ("A", "B") for f in (0.05, 0.10, 0.20)
            for r in range(REPS)]
    res = Parallel(n_jobs=8)(delayed(one)(rg, f, r) for rg, f, r in jobs)
    rows = [x for sub in res for x in sub]

    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(DATA, "e5b_count_vs_identity.csv")
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write(f"# E5b count vs identity | seed={SEED} | n={N} rank={RANK} "
                 f"p={P_OBS} reps={REPS}\n")
        fh.write("# count-matched keeps the SAME per-row count as published but picks "
                 "positions uniformly at random\n")
        fh.write("# coverage = mean retained observations per row (post-truncation)\n")
    df.to_csv(csv, mode="a", index=False)

    print()
    print("=" * 92)
    print(f"{'regime':>7s} {'frac':>5s} {'arm':>13s} {'relerr':>8s} {'se':>7s} "
          f"{'coverage':>9s} {'cov/badrow':>11s}")
    for rg in ("A", "B"):
        for f in (0.05, 0.10, 0.20):
            for arm in ("published", "count-matched", "contam-aware"):
                q = df[(df.regime == rg) & np.isclose(df.frac, f) & (df.arm == arm)]
                print(f"{rg:>7s} {f:5.2f} {arm:>13s} {q.relerr.mean():8.4f} "
                      f"{q.relerr.std(ddof=1)/np.sqrt(len(q)):7.4f} "
                      f"{q.coverage.mean():9.2f} {q.cov_bad_rows.mean():11.2f}")
            print()

    print("=" * 92)
    print("If the count dominated, count-matched should sit close to published.")
    print("=" * 92)
    print(f"{'regime':>7s} {'frac':>5s} {'published':>10s} {'count-mat':>10s} "
          f"{'contam-aw':>10s} {'|pub-cm|':>9s} {'|cm-ca|':>9s}")
    for rg in ("A", "B"):
        for f in (0.05, 0.10, 0.20):
            g = lambda a: df[(df.regime == rg) & np.isclose(df.frac, f)
                             & (df.arm == a)].relerr.mean()
            print(f"{rg:>7s} {f:5.2f} {g('published'):10.4f} {g('count-matched'):10.4f} "
                  f"{g('contam-aware'):10.4f} "
                  f"{abs(g('published')-g('count-matched')):9.4f} "
                  f"{abs(g('count-matched')-g('contam-aware')):9.4f}")
