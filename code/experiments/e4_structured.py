"""E4: structured contamination and the breakdown of the median-based statistic.

Regimes
  A  independent element noise      -- the setting of E3, should behave normally
  B  whole rows corrupted           -- the median sees a minority of bad rows
  C  whole rows + whole columns      -- both margins damaged

Question asked: does the MADN-based threshold inflate (lose its robustness) under
structured contamination, and does the relative relaxation recover any of it?
This is the empirical counterpart of Proposition 5.1.
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
REL = (0.0, 0.10, 0.30)


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
    elif regime == "C":
        rows = rng.random(n) < frac
        cols = rng.random(n) < frac
        bad = mask & (rows[:, None] | cols[None, :])
    shift = mag * sigma * rng.uniform(0.5, 1.5, (n, n))
    Y = Y + np.where(bad, shift, 0.0)
    return X, Y, mask, bad


def init_winsor(Y, mask, rank, z=3.0):
    v = Y[mask]
    s = 1.4826 * np.median(np.abs(v - np.median(v)))
    Z = np.where(mask, np.clip(Y, -z * s, z * s), 0.0)
    U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    return U[:, :rank] * np.sqrt(sv[:rank]), Vt[:rank].T * np.sqrt(sv[:rank])


def run(Y, mask, rank, lam, rel):
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
    that = np.asarray(that)
    theta_trunc = float(np.minimum.accumulate(that)[-1]) * (1.0 + rel)
    for _ in range(3):
        U, V = pbcd_step(Y, mask, U, V, theta_trunc, lam, INNER)
    return that, U, V, theta_trunc


def one(regime, frac, rep, mag=6.0, sigma=0.05):
    rng = np.random.default_rng(SEED + 7919 * rep + int(frac * 1000)
                                + {"A": 0, "B": 1, "C": 2}[regime])
    X, Y, mask, bad = make(rng, N, RANK, P_OBS, sigma, regime, frac, mag)
    lam = 1e-3 * float(np.trace(Y.T @ Y) / mask.sum())
    nX = np.linalg.norm(X)
    clean_scale = sigma
    out = []
    for rel in REL:
        that, U, V, th = run(Y, mask, RANK, lam, rel)
        relerr = float(np.linalg.norm(X - U @ V.T) / nX)
        out.append(dict(regime=regime, frac=frac, rep=rep, rel=rel, relerr=relerr,
                        madn_terminal=float(np.median(that[-20:])),
                        theta_trunc=th,
                        inflate=float(np.median(that[-20:]) / clean_scale),
                        frac_bad=float(bad.sum() / mask.sum()),
                        rows_hit=float(np.mean(bad.any(axis=1)))))
    return out


if __name__ == "__main__":
    print(f"E4  structured contamination  n={N} rank={RANK} p={P_OBS} reps={REPS}")
    jobs = [(rg, f, r) for rg in ("A", "B", "C") for f in (0.05, 0.10, 0.20)
            for r in range(REPS)]
    res = Parallel(n_jobs=8)(delayed(one)(rg, f, r) for rg, f, r in jobs)
    rows = [x for sub in res for x in sub]

    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(DATA, "e4_structured.csv")
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write(f"# E4 structured contamination | seed={SEED} | n={N} rank={RANK} "
                 f"p={P_OBS} reps={REPS}\n")
        fh.write("# regime A = independent elements, B = whole rows, "
                 "C = whole rows + whole columns\n")
        fh.write("# inflate = terminal MADN / sigma (1.0 would mean the statistic "
                 "still tracks the clean noise scale)\n")
    df.to_csv(csv, mode="a", index=False)

    print()
    print("=" * 90)
    print(f"{'regime':>7s} {'frac':>5s} {'rel':>5s} {'relerr':>8s} {'se':>7s} "
          f"{'inflate':>8s} {'theta_tr':>9s} {'bad%':>6s} {'rows hit':>9s}")
    for rg in ("A", "B", "C"):
        for f in (0.05, 0.10, 0.20):
            s = df[(df.regime == rg) & (np.isclose(df.frac, f))]
            for rel in REL:
                q = s[np.isclose(s.rel, rel)]
                print(f"{rg:>7s} {f:5.2f} {rel:5.2f} {q.relerr.mean():8.4f} "
                      f"{q.relerr.std(ddof=1)/np.sqrt(len(q)):7.4f} "
                      f"{q.inflate.mean():8.2f} {q.theta_trunc.mean():9.4f} "
                      f"{100*q.frac_bad.mean():6.1f} {q.rows_hit.mean():9.2f}")
            print()

    print("=" * 90)
    print("MADN inflation: A vs B vs C at the same nominal contamination fraction")
    print("=" * 90)
    print(f"{'frac':>5s} " + " ".join(f"{'regime '+r:>12s}" for r in "ABC"))
    for f in (0.05, 0.10, 0.20):
        cells = []
        for rg in "ABC":
            q = df[(df.regime == rg) & np.isclose(df.frac, f) & np.isclose(df.rel, 0.0)]
            cells.append(f"{q.inflate.mean():12.2f}")
        print(f"{f:5.2f} " + " ".join(cells))
