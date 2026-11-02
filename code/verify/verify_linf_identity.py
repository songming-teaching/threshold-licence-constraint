"""Does the min rule coincide with any L_p isotonic projection? Quantitative answer.

  L_inf isotonic error  = (1/2) * max drawup      (LP, always)
  min-rule sup deviation =        max drawup
  ratio = 2 exactly.
"""
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(60601)


def iso_linf_err(th):
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
    return float(res.fun)


errs, half, ratio, dev = [], [], [], []
for _ in range(800):
    K = int(rng.integers(3, 26))
    t = rng.standard_normal(K)
    draw = float(np.max(np.maximum.accumulate(t) - t))
    ys = iso_linf_err(t)
    dm = float(np.max(np.maximum.accumulate(t) - t))   # same quantity: min rule deviation
    dm = float(np.max(t[:np.argmin(np.minimum.accumulate(t)) + 1] - np.minimum.accumulate(t))) \
        if False else float(np.max(np.abs(np.minimum.accumulate(t) - t)))
    errs.append(ys); half.append(draw / 2.0); dev.append(dm)
errs, half, dev = np.array(errs), np.array(half), np.array(dev)
print(f"LP Linf isotonic error vs drawup/2:  max|diff| = {np.max(np.abs(errs-half)):.2e}")
print(f"  relative: max = {np.max(np.abs(errs-half)/np.maximum(half,1e-12)):.2e}")
print(f"min-rule sup deviation / LP error:    min = {np.min(dev/errs):.10f}, "
      f"max = {np.max(dev/errs):.10f}")
print(f"min-rule sup deviation vs drawup:     max|diff| = {np.max(np.abs(dev-np.array([0.0])))}"
      f"  (printed below properly)")
draws = 2 * half
print(f"  min-rule deviation vs drawup:       max|diff| = {np.max(np.abs(dev-draws)):.2e}")
print()
print("  counterexample theta_hat = (0, 3, 1, 2):")
t = np.array([0.0, 3.0, 1.0, 2.0])
print(f"    min rule      {np.minimum.accumulate(t)}")
print(f"    L_inf iso err {iso_linf_err(t):.4f}, drawup {np.max(np.maximum.accumulate(t)-t):.4f}")
