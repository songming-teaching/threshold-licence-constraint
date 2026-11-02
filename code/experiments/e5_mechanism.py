"""E5: confirm or refute the "information loss" mechanism behind E4.

E4 showed structured (whole-row) contamination destroys reconstruction 3-5x while
leaving the MADN statistic untouched. The proposed mechanism was: the corrupted
rows have all their observations truncated, so their factor has no data.

Three tests, in increasing strength:

  T1  DIRECT   count inliers per row, split by contaminated / clean.
               Mechanism predicts contaminated rows keep ~0 inliers.
  T2  CONTROL  disable truncation entirely (theta = inf) under regime B.
               If the loss is caused by data being cut, restoring the data must
               recover; if the loss is caused by bad data entering the fit, it
               must get worse. Either answer discriminates.
  T3  ORACLE   give the solver the true bad mask and drop only the corrupted
               entries, keeping every clean observation of a corrupted row.
               This is the best any threshold rule could do; the gap between
               T3 and the real rule is the price of not knowing which is which.
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
    shift = mag * sigma * rng.uniform(0.5, 1.5, (n, n))
    Y = Y + np.where(bad, shift, 0.0)
    return X, Y, mask, bad


def init_winsor(Y, mask, rank, z=3.0):
    v = Y[mask]
    s = 1.4826 * np.median(np.abs(v - np.median(v)))
    Z = np.where(mask, np.clip(Y, -z * s, z * s), 0.0)
    U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    return U[:, :rank] * np.sqrt(sv[:rank]), Vt[:rank].T * np.sqrt(sv[:rank])


def inlier_mask(Y, mask, U, V, theta):
    R = Y - U @ V.T
    return mask & (np.abs(R) <= theta)


def anneal(Y, mask, rank, lam, force_mask=None):
    """force_mask: if given, overrides the inlier set every sweep (T2/T3)."""
    U, V = init_winsor(Y, mask, rank)
    cur = 1e18
    for _ in range(5):
        U, V = pbcd_step(Y, mask if force_mask is None else force_mask,
                         U, V, cur, lam, INNER)
    that = []
    for _ in range(K):
        for _ in range(INNER):
            U, V = pbcd_step(Y, mask if force_mask is None else force_mask,
                             U, V, cur, lam)
        R = (Y - U @ V.T)[mask]
        h = madn(R)
        cur = min(h, cur)
        that.append(h)
    return np.asarray(that), U, V


def one(regime, frac, rep, mag=6.0, sigma=0.05):
    rng = np.random.default_rng(SEED + 7919 * rep + int(frac * 1000)
                                + {"A": 0, "B": 1}[regime])
    X, Y, mask, bad = make(rng, N, RANK, P_OBS, sigma, regime, frac, mag)
    lam = 1e-3 * float(np.trace(Y.T @ Y) / mask.sum())
    nX = np.linalg.norm(X)
    out = []

    # ---------------- T1: inliers per row at the published schedule
    that, U, V = anneal(Y, mask, RANK, lam)
    theta = float(np.minimum.accumulate(that)[-1])
    W = inlier_mask(Y, mask, U, V, theta)
    bad_rows = bad.any(axis=1)
    inl = W.sum(axis=1)
    inl_bad = inl[bad_rows] if bad_rows.any() else np.array([np.nan])
    inl_ok = inl[~bad_rows] if (~bad_rows).any() else np.array([np.nan])
    out.append(dict(regime=regime, frac=frac, rep=rep, arm="published",
                    relerr=float(np.linalg.norm(X - U @ V.T) / nX),
                    inl_bad_rows=float(np.mean(inl_bad)),
                    inl_clean_rows=float(np.mean(inl_ok)),
                    frac_rows_starved=float(np.mean(inl_bad < RANK + 1))
                    if bad_rows.any() else np.nan,
                    theta=theta))

    # ---------------- T2: no truncation at all
    U, V = init_winsor(Y, mask, RANK)
    for _ in range(5):
        U_, V_ = pbcd_step(Y, mask, U, V, np.inf, lam)
        U, V = U_, V_
    for _ in range(K):
        U, V = pbcd_step(Y, mask, U, V, np.inf, lam)
    out.append(dict(regime=regime, frac=frac, rep=rep, arm="no-truncation",
                    relerr=float(np.linalg.norm(X - U @ V.T) / nX),
                    inl_bad_rows=float(mask[bad_rows].sum() / max(bad_rows.sum(), 1)),
                    inl_clean_rows=float(mask[~bad_rows].sum() / max((~bad_rows).sum(), 1)),
                    frac_rows_starved=np.nan, theta=np.inf))

    # ---------------- T3: oracle mask (drop exactly the corrupted entries)
    oracle = mask & ~bad
    that3, U, V = anneal(Y, mask, RANK, lam, force_mask=oracle)
    theta3 = float(np.minimum.accumulate(that3)[-1])
    W3 = oracle
    out.append(dict(regime=regime, frac=frac, rep=rep, arm="oracle-mask",
                    relerr=float(np.linalg.norm(X - U @ V.T) / nX),
                    inl_bad_rows=float(np.mean(W3[bad_rows].sum(axis=1)))
                    if bad_rows.any() else np.nan,
                    inl_clean_rows=float(np.mean(W3[~bad_rows].sum(axis=1))),
                    frac_rows_starved=0.0, theta=theta3))
    return out


if __name__ == "__main__":
    print(f"E5 mechanism confirmation  n={N} rank={RANK} p={P_OBS} reps={REPS}")
    jobs = [(rg, f, r) for rg in ("A", "B") for f in (0.05, 0.10, 0.20)
            for r in range(REPS)]
    res = Parallel(n_jobs=8)(delayed(one)(rg, f, r) for rg, f, r in jobs)
    rows = [x for sub in res for x in sub]

    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(DATA, "e5_mechanism.csv")
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write(f"# E5 mechanism confirmation | seed={SEED} | n={N} rank={RANK} "
                 f"p={P_OBS} reps={REPS}\n")
        fh.write("# arm: published = MADN+min rule; no-truncation = theta=inf; "
                 "oracle-mask = corrupted entries removed perfectly\n")
        fh.write("# inl_bad_rows / inl_clean_rows = mean number of usable "
                 "observations per row\n")
    df.to_csv(csv, mode="a", index=False)

    print()
    print("=" * 96)
    print(f"{'regime':>7s} {'frac':>5s} {'arm':>14s} {'relerr':>8s} {'se':>7s} "
          f"{'inl/badrow':>11s} {'inl/cleanrow':>13s} {'starved':>8s}")
    for rg in ("A", "B"):
        for f in (0.05, 0.10, 0.20):
            for arm in ("published", "no-truncation", "oracle-mask"):
                q = df[(df.regime == rg) & np.isclose(df.frac, f) & (df.arm == arm)]
                print(f"{rg:>7s} {f:5.2f} {arm:>14s} {q.relerr.mean():8.4f} "
                      f"{q.relerr.std(ddof=1)/np.sqrt(len(q)):7.4f} "
                      f"{q.inl_bad_rows.mean():11.2f} "
                      f"{q.inl_clean_rows.mean():13.2f} "
                      f"{q.frac_rows_starved.mean():8.3f}")
            print()

    print("=" * 96)
    print("The discriminating comparison (regime B, relerr)")
    print("=" * 96)
    print(f"{'frac':>5s} {'published':>10s} {'no-trunc':>10s} {'oracle':>10s} "
          f"{'pub-oracle':>11s}")
    for f in (0.05, 0.10, 0.20):
        g = lambda arm: df[(df.regime == "B") & np.isclose(df.frac, f)
                           & (df.arm == arm)].relerr.mean()
        print(f"{f:5.2f} {g('published'):10.4f} {g('no-truncation'):10.4f} "
              f"{g('oracle-mask'):10.4f} {g('published')-g('oracle-mask'):+11.4f}")
