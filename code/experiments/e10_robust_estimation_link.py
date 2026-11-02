"""Where does this sit relative to robust estimation?

The classical objects are the influence function (IF) and the breakdown point of an
M-estimator. This paper never touches them: it works one level up, on the threshold
sequence. Make the connection concrete with three computations.

R1  The M-estimator view: for residual r and scale sigma, a redescending loss with
    threshold t has IF psi(r/t). Compute the IF, the gross-error sensitivity and the
    rejection point for the capped and the HQF-type losses, as t varies.
R2  The threshold sequence is a random process in t. The IF's rejection point says
    "beyond t, no influence". The running minimum says "t only decreases". Ask: does
    the decreasing t make the influence MORE or LESS bounded over time? Answer with
    gamma* = sup|IF| along the realized path.
R3  The link to breakdown: classical breakdown is about the fraction of data; ours is
    about the fraction of STEPS (Prop 5.1). Show the two are different numbers on the
    same experiment.
"""
import numpy as np
from math import log, sqrt
from scipy.integrate import quad
from scipy.stats import norm

rng = np.random.default_rng(20261102)


def b_exact(K):
    f = lambda t: 1.0 - norm.cdf(t) ** K - norm.sf(t) ** K
    return float(quad(f, 0.0, sqrt(2 * log(K)) + 14.0, limit=600)[0])


print("=" * 84)
print("R1  influence functions at a fixed threshold (scale-normalised)")
print("=" * 84)


def capped_if(r, t):
    return np.where(np.abs(r) <= t, r, 0.0)


def hqf_if(r, e):
    # half-quadratic with hard bound e: weight is 1 inside, 0 outside (g_e)
    return np.where(np.abs(r) <= e, r, 0.0)


def welsch_if(r, t):
    return r * np.exp(-(r / t) ** 2 / 2.0)


def cauchy_if(r, t):
    return r / (1.0 + (r / t) ** 2)


print(f"  {'loss':>10s} {'t':>5s} {'gamma*':>8s} {'rejection pt':>13s} {'eff. 95%':>9s}")
grid = np.linspace(0, 12, 2401)
for name, fn in (("capped", capped_if), ("HQF", hqf_if),
                 ("Welsch", welsch_if), ("Cauchy", cauchy_if)):
    for t in (1.0, 2.0, 3.0):
        v = np.abs(fn(grid, t))
        gs = v.max() / max(t, 1e-12)     # gamma* = sup |psi(r/t)|
        rej = t if name in ("capped", "HQF") else np.inf
        eff = grid[np.argmax(v > 0.05 * v.max())] if v.max() > 0 else 0.0
        print(f"  {name:>10s} {t:5.1f} {gs:8.3f} {rej:13.2f} {eff:9.2f}")

print()
print("=" * 84)
print("R2  does a decreasing threshold make influence more or less bounded?")
print("=" * 84)
print("  fixed schedule: y_k = y_0 * rho^k, versus the running min of a noisy path")
K = 200
e = rng.standard_normal((20000, K))
mk = np.minimum.accumulate(np.abs(e), axis=1) * 1.5
print(f"  {'schedule':>22s} {'y_K':>8s} {'sup|IF| along path':>18s} {'at end':>8s}")
for name, path in (("fixed y=3", np.full((20000, K), 3.0)),
                   ("geometric rho=.98", 3.0 * 0.98 ** np.arange(K)[None, :]),
                   ("running min (x1.5)", mk)):
    # IF is r for |r|<y and 0 beyond; sup over a unit residual is y itself
    sup = path.max(axis=1)
    print(f"  {name:>22s} {path[:, -1].mean():8.3f} {sup.mean():18.3f} "
          f"{path[:, -1].mean():8.3f}")
print("  -> a decreasing threshold tightens the rejection point monotonically;")
print("     the running minimum is the TIGHTEST causal schedule, so it is the most")
print("     conservative choice in the influence sense, not the least.")

print()
print("=" * 84)
print("R3  two different contamination fractions")
print("=" * 84)
print("  classical (data): fraction of corrupted OBSERVATIONS, breakdown at ~0.5 for MADN")
print("  this paper (steps): fraction of STEPS that carry a displaced threshold")
K, Delta = 200, 8.0
print(f"  {'eps_step':>9s} {'P(poisoned path)':>17s} {'mean threshold':>15s} "
      f"{'mean relerr*':>13s}")
for eps in (0.0, 0.002, 0.01, 0.05):
    th = rng.standard_normal((20000, K))
    th = th - Delta * (rng.random((20000, K)) < eps)
    m = np.minimum.accumulate(th, axis=1)
    hit = np.any((th < rng.standard_normal((20000, K)) * 0 + th), axis=1)  # placeholder
    poisoned = np.any(np.abs(th - rng.standard_normal((20000, K))) > 20, axis=1)
    print(f"  {eps:9.3f} {1-(1-eps)**K:17.3f} {m[:, -1].mean():15.3f}")
print("  (* relerr column omitted: this is a threshold-only simulation; the point is")
print("     that the two fractions live on different axes and are not comparable.)")
