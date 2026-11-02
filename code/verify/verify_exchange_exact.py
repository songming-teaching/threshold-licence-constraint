"""Reproduce the exchange table of section 3.3 exactly (seed 112358, K=40, 20000 paths)."""
import json
import os

import numpy as np

rng = np.random.default_rng(112358)
K, reps = 40, 20000
th = rng.standard_normal((reps, K))
m = th.min(axis=1)
bK = float(-m.mean())
rows = []
print(f"{'delta':>7s} {'bias':>9s} {'drop':>9s} {'excess':>9s} {'sum':>9s}")
for d in (0.0, 0.1, 0.5, 1.0, 2.0, 3.0):
    y = np.maximum(np.minimum.accumulate(th, axis=1), (m + d)[:, None])
    V = np.max(np.maximum(y - th, 0.0), axis=1)
    bias = float(-y[:, -1].mean())
    rows.append(dict(delta=d, bias=bias, drop=bK - bias, excess=float(V.mean()),
                     sum=bias + float(V.mean())))
    print(f"{d:7.2f} {bias:9.4f} {bK-bias:9.4f} {V.mean():9.4f} "
          f"{bias+V.mean():9.4f}")

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "results", "exchange.json"), "w", encoding="utf-8") as fh:
    json.dump(dict(seed=112358, K=K, reps=reps, bK=bK, rows=rows), fh, indent=2)
print()
print("written: results/exchange.json")
