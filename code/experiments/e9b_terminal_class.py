"""Correcting my own previous claim.

Prop 2.2 forces y_k <= m_k for EVERY causal non-increasing y, and m_K <= thetahat_K.
So the terminal licence y_K <= thetahat_K is IMPLIED, not a relaxation.
min(mean, thetahat_K) is not non-increasing, so it was never a counterexample.

Question that remains: is the terminal-licence class strictly larger than the pathwise
class, and does it contain anything better than the running minimum?
"""
import numpy as np

rng = np.random.default_rng(1)


def b_exact(K):
    from math import log, sqrt
    from scipy.integrate import quad
    from scipy.stats import norm
    f = lambda t: 1.0 - norm.cdf(t) ** K - norm.sf(t) ** K
    return float(quad(f, 0.0, sqrt(2 * log(K)) + 14.0, limit=600)[0])


K, reps = 50, 200000
e = rng.standard_normal((reps, K))
m = np.minimum.accumulate(e, axis=1)

print("=" * 84)
print("1. Pathwise class A vs terminal class B")
print("=" * 84)
print("   A = {causal, non-increasing, y_k <= thetahat_k for all k}")
print("   B = {causal, non-increasing, y_K <= thetahat_K only}")
print("   Prop 2.2 => A is contained in B with y_k <= m_k, and m_K <= thetahat_K, so B's extra")
print("   condition is inactive ON A. Is B strictly larger?")

# A realisable element of B that is not in A: y_k = m_k for k<K,
# then lift the last value as far as monotonicity and the terminal licence allow.
yK = np.minimum(m[:, -2], e[:, -1])
cand = np.concatenate([m[:, :-1], yK[:, None]], axis=1)
mono = np.all(np.diff(cand, axis=1) <= 1e-12, axis=1)
term = np.all(yK <= e[:, -1] + 1e-12)
print(f"   lift-last rule: non-increasing on {mono.mean():.4f} of paths, "
      f"terminal licence on {term.mean():.4f}")
print(f"   exceeds m_K on {np.mean(yK > m[:, -1]):.4f} of paths -> not in A")

print()
print("=" * 84)
print("2. Does that element beat the running minimum?")
print("=" * 84)
print(f"   {'rule':>28s} {'MSE':>9s} {'E[y_K]':>9s}")
print(f"   {'running min (in A)':>28s} {np.mean(m[:, -1] ** 2):9.4f} {m[:, -1].mean():9.4f}")
print(f"   {'lift-last (in B only)':>28s} {np.mean(cand[:, -1] ** 2):9.4f} "
      f"{cand[:, -1].mean():9.4f}")
print(f"   {'running min + b_K':>28s} {np.mean((m[:, -1] + b_exact(K)) ** 2):9.4f} "
      f"{(m[:, -1] + b_exact(K)).mean():9.4f}")

print()
print("=" * 84)
print("3. What actually relaxes the class?  Dropping monotonicity, not the licence.")
print("=" * 84)
print(f"   {'rule':>28s} {'MSE':>9s} {'monotone':>9s} {'licence':>8s}")
rules = {
    "running min (A)": m[:, -1],
    "running min + b_K (not A)": m[:, -1] + b_exact(K),
    "mean (not monotone)": e.mean(axis=1),
    "thetahat_K": e[:, -1],
}
for name, v in rules.items():
    mono_ok = "yes" if name in ("running min (A)", "thetahat_K") else "no"
    print(f"   {name:>28s} {np.mean(v ** 2):9.4f} {mono_ok:>9s} "
          f"{np.mean(v <= e[:, -1] + 1e-12):8.3f}")
