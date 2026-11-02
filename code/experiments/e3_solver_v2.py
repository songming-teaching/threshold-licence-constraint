"""E3 v2: solver-level comparison, with the truncation threshold and the reported
threshold kept separate.

Lesson from v1 (see logs/e3_日志_v1_作废.md): shifting the TRUNCATION threshold by a
multiple of b_K destroys the robust mechanism, because b_K is measured in units of
the estimator's own spread, which is tiny in a converging solver. The shift that
matters operationally is relative to the threshold itself.

Two objects are therefore tracked apart:
  theta_trunc : what the state update actually cuts at
  theta_report: what the frontier prices
and the experiment asks two separate questions:
  Q1  does relaxing theta_trunc by a RELATIVE amount change the reconstruction?
  Q2  does the (bias, over-trust) pair of the reported schedule sit where the
      theory says it should?
"""
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed
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
REL = (0.0, 0.05, 0.10, 0.20, 0.30)


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


def run_path(Y, mask, rank, lam, scale=1.0):
    """Anneal with theta_trunc = scale * (running minimum of the MADN path)."""
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
        cur = min(h, cur) * scale if scale != 1.0 else min(h, cur)
        that.append(h)
    return np.asarray(that), U, V


def one(eps, rep, mag=6.0, sigma=0.05):
    rng = np.random.default_rng(SEED + 1000 * rep + int(eps * 1000))
    X, Y, mask, bad = make(rng, N, RANK, P_OBS, sigma, eps, mag)
    lam = 1e-3 * float(np.trace(Y.T @ Y) / mask.sum())
    nX = np.linalg.norm(X)
    out = []
    for rel in REL:
        that, U, V = run_path(Y, mask, RANK, lam, scale=1.0 + rel)
        # settle the state at the FINAL truncation threshold so the comparison is
        # at equal footing across schedules
        theta_trunc = float(np.minimum.accumulate(that)[-1]) * (1.0 + rel)
        for _ in range(3):
            U, V = pbcd_step(Y, mask, U, V, theta_trunc, lam, INNER)
        relerr = float(np.linalg.norm(X - U @ V.T) / nX)
        mk = np.minimum.accumulate(that)
        trunc_path = mk * (1.0 + rel)              # truncation threshold actually used
        out.append(dict(eps=eps, rep=rep, rel=rel, relerr=relerr,
                        theta_trunc_final=theta_trunc,
                        theta_trunc_mean=float(trunc_path.mean()),
                        that_mean=float(that.mean()), that_last=float(that[-1]),
                        madn_terminal=float(np.median(that[-20:])),
                        n_bad=int(bad.sum()), frac_bad=float(bad.sum()/mask.sum())))
    return out


if __name__ == "__main__":
    print(f"E3 v2  n={N} rank={RANK} p={P_OBS} K={K} reps={REPS}")
    print(f"       relative truncation relaxation {REL}")
    jobs = [(e, r) for e in EPS_LIST for r in range(REPS)]
    res = Parallel(n_jobs=8)(delayed(one)(e, r) for e, r in jobs)
    rows = [x for sub in res for x in sub]

    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(DATA, "e3_solver_v2.csv")
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write(f"# E3 v2 | seed={SEED} | n={N} rank={RANK} p={P_OBS} K={K} reps={REPS}\n")
        fh.write("# rel = relative relaxation of the TRUNCATION threshold; 0 = published rule\n")
        fh.write("# relerr measured after 3 proximal sweeps at the final truncation threshold\n")
    df.to_csv(csv, mode="a", index=False)

    print()
    print("=" * 88)
    print(f"{'eps':>5s} {'rel':>6s} {'relerr':>8s} {'se':>7s} {'theta_trunc':>12s} "
          f"{'madn_term':>10s} {'ratio':>7s}")
    for eps in EPS_LIST:
        s = df[df.eps == eps]
        for rel in REL:
            q = s[s.rel == rel]
            print(f"{eps:5.2f} {rel:6.2f} {q.relerr.mean():8.4f} "
                  f"{q.relerr.std(ddof=1)/np.sqrt(len(q)):7.4f} "
                  f"{q.theta_trunc_final.mean():12.4f} "
                  f"{q.madn_terminal.mean():10.4f} "
                  f"{(q.theta_trunc_final/q.madn_terminal).mean():7.3f}")
        print()

    print("=" * 88)
    print("paired against rel=0 (positive = relaxing makes it worse)")
    print("=" * 88)
    for eps in EPS_LIST:
        s = df[df.eps == eps]
        base = s[s.rel == 0.0].set_index("rep").relerr
        for rel in REL[1:]:
            q = s[s.rel == rel].set_index("rep").relerr
            d = base - q
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else 0.0
            print(f"  eps={eps:5.2f} rel={rel:4.2f}: {d.mean():+.5f}  "
                  f"se {d.std(ddof=1)/np.sqrt(len(d)):.5f}  t {t:+6.1f}")
        print()
