"""Re-run the Prop 5.2 table with the CORRECT floor value.

The published table used floor = -2.610, which is the Gumbel centring constant
sqrt(2 ln K) - (log log K + log 4pi)/(2 sqrt(2 ln K)), NOT b_200.
Exact b_200 = 2.7460. Re-run with the correct floor.
"""
import json
import os
from math import log, pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

SEED = 20261102
rng = np.random.default_rng(SEED)


def b_exact(K):
    f = lambda t: 1.0 - norm.cdf(t) ** K - norm.sf(t) ** K
    return float(quad(f, 0.0, sqrt(2 * log(K)) + 14.0, limit=600)[0])


K, Delta, reps = 200, 8.0, 20000
bK = b_exact(K)
print(f"b_200 exact = {bK:.4f}")
print(f"Gumbel centring (what the old table used) = "
      f"{sqrt(2*log(K)) - (log(log(K))+log(4*pi))/(2*sqrt(2*log(K))):.4f}")

print()
print("=" * 92)
print(f"{'eps':>6s} {'min rule':>10s} {'floored':>10s} {'floored bias':>13s} "
      f"{'b_K':>8s} {'over-trust':>11s} {'old table':>10s}")
rows = []
for eps in (0.0, 0.002, 0.01, 0.05):
    th = rng.standard_normal((reps, K))
    bad = rng.random((reps, K)) < eps
    th = th - Delta * bad
    rm = np.minimum.accumulate(th, axis=1)
    y = np.maximum(rm, -bK)
    bias = 0.0 - float(np.mean(y[:, -1]))
    ot = float(np.mean(np.max(np.maximum(y - th, 0.0), axis=1)))
    rows.append(dict(eps=eps, min_mean=float(rm[:, -1].mean()),
                     floored_mean=float(y[:, -1].mean()),
                     floored_bias=bias, bK=bK, overtrust=ot))
    print(f"{eps:6.3f} {rm[:, -1].mean():10.4f} {y[:, -1].mean():10.4f} "
          f"{bias:13.4f} {bK:8.4f} {ot:11.4f}")

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(os.path.dirname(here), "data", "e12_breakdown_corrected.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(dict(seed=SEED, K=K, Delta=Delta, reps=reps, bK=bK, rows=rows),
              fh, indent=2)
print()
print("written:", out)
