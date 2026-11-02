"""Cross-check every number in the draft against the stored results.

Each entry: (value as printed in the draft, value as recomputed). A mismatch is an
error in the text, not in the arithmetic.

Revision 3 additions: Theorem 4.5 table, the regenerated 4.3/4.5 columns, the
Gumbel expansion-error column, and the heavy-tail asymptotic-ratio sequence.
"""
import json
import os
from math import gamma, pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))


def b_exact(K):
    f = lambda t: 1.0 - norm.cdf(t) ** K - norm.sf(t) ** K
    return float(quad(f, 0.0, sqrt(2 * np.log(K)) + 14.0, limit=600,
                      epsabs=1e-12, epsrel=1e-12)[0])


def b_t(nu, K):
    scale = sqrt((nu - 2.0) / nu)

    def g(s):
        x = np.exp(s)
        F = float(tdist.cdf(x / scale, nu))
        if F >= 1.0:
            return 0.0
        return (1.0 - F ** K - (1.0 - F) ** K) * x
    return float(quad(g, -40.0, np.log(1e7), limit=500, epsabs=1e-11)[0])


def b_moving(K, drop):
    a = np.linspace(drop, 0.0, K)

    def g(s):
        lo = 1.0 - np.exp(np.sum(np.log1p(-norm.cdf(-s - a))))
        with np.errstate(divide="ignore"):
            lg = np.sum(np.log1p(-norm.cdf(s - a)))
        return lo - (np.exp(lg) if lg > -700 else 0.0)
    return float(quad(g, 0.0, drop + 12.0, limit=400, epsabs=1e-11)[0])


core = json.load(open(os.path.join(HERE, "results", "core.json"), encoding="utf-8"))
fin = json.load(open(os.path.join(HERE, "results", "final.json"), encoding="utf-8"))
clos = json.load(open(os.path.join(HERE, "results", "closing.json"), encoding="utf-8"))
sol = json.load(open(os.path.join(HERE, "results", "solver_detrended.json"), encoding="utf-8"))
reg = json.load(open(os.path.join(HERE, "results", "regimes.json"), encoding="utf-8"))
regen = json.load(open(os.path.join(HERE, "results", "regen.json"), encoding="utf-8"))

checks = []


def chk(claim, printed, actual, tol):
    ok = abs(printed - actual) <= tol
    checks.append((ok, claim, printed, actual))
    return ok


# ---- section 2
ce = core["counterexample"]
chk("2.3 counterexample first entry", 0.0, float(ce["th"][0]), 1e-12)
chk("2.4 L-inf factor of two (median ratio)", 2.0, core["linf_ratio"]["median"], 1e-9)
chk("2.4 L-inf ratio q10", 2.0, core["linf_ratio"]["q10"], 1e-9)
chk("2.4 L-inf ratio q90", 2.0, core["linf_ratio"]["q90"], 1e-9)
chk("2.2 maximality escape", 0.0, core["maximality_escape"], 1e-12)

# ---- section 3, frontier table
want = {10: ("1.5415", "0.0000", "1.3752", "1.3709", "1.2046", "0.5365", "1.3881", "0.3832"),
        40: ("2.1646", "0.0000", "2.0027", "2.0440", "1.8821", "0.5665", "2.1030", "0.5050"),
        150: ("2.6464", "0.0000", "2.5455", "2.5504", "2.4495", "0.6017", "2.6150", "0.5702")}
for row in core["frontier"]:
    K = row["K"]
    d = {r["schedule"]: r for r in row["rows"]}
    chk(f"3.4 K={K} b_K", float(want[K][0]), row["bK"], 5e-4)
    chk(f"3.4 K={K} min overtrust", float(want[K][1]), d["running min"]["excess"], 5e-4)
    chk(f"3.4 K={K} Linf bias", float(want[K][2]), d["Linf iso"]["bias"], 5e-4)
    chk(f"3.4 K={K} Linf overtrust", float(want[K][3]), d["Linf iso"]["excess"], 5e-4)
    chk(f"3.4 K={K} Linf excess", float(want[K][4]), d["Linf iso"]["sum"] - row["bK"], 5e-4)
    chk(f"3.4 K={K} L2 bias", float(want[K][5]), d["L2 iso"]["bias"], 5e-4)
    chk(f"3.4 K={K} L2 overtrust", float(want[K][6]), d["L2 iso"]["excess"], 5e-4)
    chk(f"3.4 K={K} L2 excess", float(want[K][7]), d["L2 iso"]["sum"] - row["bK"], 5e-4)

# ---- section 3, exchange table
exch = core["exchange"]
ex_want = [(0.0, "2.1644"), (0.1, "2.0644"), (0.5, "1.6644"), (1.0, "1.1644"),
           (2.0, "0.1644"), (3.0, "-0.8356")]
for (d, bias) in ex_want:
    row = [r for r in exch if abs(r["delta"] - d) < 1e-12][0]
    chk(f"3.3 delta={d} bias", float(bias), row["bias"], 5e-4)
    chk(f"3.3 delta={d} sum constant", 2.1644, row["sum"], 5e-4)

# ---- section 4.1 / 4.2
chk("4.2 b_2 quadrature", 0.56418958, b_exact(2), 1e-8)
chk("4.2 b_2 simulated", 0.5640, core["floor_b2"], 5e-5)
gumbel_want = {10: ("1.5388", "2.1460", "1.3619", "1.6309", "6.0"),
               100: ("2.5076", "3.0349", "2.3663", "2.5565", "1.95"),
               1000: ("3.2414", "3.7169", "3.1165", "3.2718", "0.94"),
               10000: ("3.8516", "4.2919", "3.7384", "3.8729", "0.55"),
               100000: ("4.3843", "4.7985", "4.2802", "4.4005", "0.37")}
for K, (ex_, sq, ng, wg, err) in gumbel_want.items():
    chk(f"4.2 K={K} exact", float(ex_), b_exact(K), 5e-4)
    a = sqrt(2 * np.log(K))
    chk(f"4.2 K={K} sqrt2lnK", float(sq), a, 5e-4)
    bc = a - (np.log(np.log(K)) + np.log(4 * pi)) / (2 * a)
    chk(f"4.2 K={K} centring", float(ng), bc, 5e-4)
    mod = bc + 0.5772156649015329 / a
    chk(f"4.2 K={K} with-gamma", float(wg), mod, 5e-4)
    chk(f"4.2 K={K} expansion error pct", float(err),
        100 * abs(mod - b_exact(K)) / b_exact(K), 0.02)

# ---- section 4.3, regenerated columns
heavy_sim = {(3, 100): 3.6164, (3, 1000): 8.0106, (3, 10000): 17.362,
             (4, 100): 3.3365, (4, 1000): 6.2665, (4, 10000): 11.364,
             (6, 100): 3.0214, (6, 1000): 4.8765, (6, 10000): 7.4468}
heavy_exact = {(3, 100): 3.6219, (3, 1000): 8.0176, (3, 10000): 17.374,
               (4, 100): 3.3439, (4, 1000): 6.2675, (4, 10000): 11.323,
               (6, 100): 3.0208, (6, 1000): 4.8788, (6, 10000): 7.4487}
ratio_want = {3: {100: 0.966, 1000: 0.993, 10000: 0.998, 100000: 1.000},
              4: {100: 0.927, 1000: 0.977, 10000: 0.993, 100000: 0.998},
              6: {100: 0.846, 1000: 0.931, 10000: 0.969, 100000: 0.986}}
for row in regen["sec43"]:
    key = (row["nu"], row["K"])
    chk(f"4.3 sim nu={key[0]} K={key[1]}", heavy_sim[key], row["sim"], 1e-3)
    chk(f"4.3 exact nu={key[0]} K={key[1]}", heavy_exact[key], row["exact"],
        max(5e-3, 5e-5 * heavy_exact[key]))
for nu in (3, 4, 6):
    C_std = gamma((nu + 1) / 2.0) * nu ** (nu / 2.0 - 1.0) / (sqrt(pi) * gamma(nu / 2.0))
    C = C_std * ((nu - 2.0) / nu) ** (nu / 2.0)
    for K in (100, 1000, 10000, 100000):
        asym = (K * C) ** (1.0 / nu) * gamma(1.0 - 1.0 / nu)
        chk(f"4.3 ratio nu={nu} K={K}", ratio_want[nu][K], b_t(nu, K) / asym, 2e-3)

# ---- section 4.4 multi
multi_want = {(1, 100): 2.5091, (10, 100): 3.2409, (100, 100): 3.8459,
              (1, 1000): 3.2488, (10, 1000): 3.8384, (100, 1000): 4.3813}
for r in core["multi"]:
    key = (r["d"], r["K"])
    if key in multi_want:
        chk(f"4.4 d={key[0]} K={key[1]}", multi_want[key], r["bias"], 6e-4)

# ---- section 4.5, Theorem 4.5 table
t45 = {0.0: ("3.2414", "3.2404", "0.0006", "3.241", 1000),
       2.0: ("2.6642", "2.6635", "0.0006", "2.353", 166),
       8.0: ("2.2082", "2.2074", "0.0007", "1.856", 46),
       16.0: ("1.9577", "1.9565", "0.0007", "1.579", 24),
       64.0: ("1.4042", "1.4035", "0.0009", "0.942", 9),
       512.0: ("0.4843", "0.4857", "0.0012", "0.000", 1)}
for row in regen["sec45"]:
    d = row["drop"]
    ex_, sim_, se_, bnd_, n_ = t45[d]
    chk(f"4.5 drop={d} exact", float(ex_), b_moving(1000, d), 5e-4)
    chk(f"4.5 drop={d} simulated", float(sim_), row["sim"], 1e-3)
    chk(f"4.5 drop={d} se", float(se_), row["se"], 2e-4)
    z = abs(row["sim"] - row["exact"]) / row["se"]
    chk(f"4.5 drop={d} |z| <= 1.9", max(z, 0.0) <= 1.9, True, 0.0)
    checks[-1] = (max(z, 0.0) <= 1.9, f"4.5 drop={d} |z| <= 1.9 (z={z:.2f})",
                  max(z, 0.0), 1.9)
chk("4.5 max row discrepancy", 0.0014,
    max(abs(r["sim"] - r["exact"]) for r in regen["sec45"]), 2e-4)

# Remark 4.7 implicit equation
for D, nstar in ((2.0, 157.2), (8.0, 45.3), (16.0, 24.7), (64.0, 7.7)):
    lo, hi = 1.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * sqrt(2 * np.log(mid)) - 1000.0 / D < 0:
            lo = mid
        else:
            hi = mid
    chk(f"4.5 n* D={D}", nstar, 0.5 * (lo + hi), 0.1)

# ---- section 5
bd_want = {0.0: (-2.749, -2.519, 1.748), 0.002: (-4.505, -2.549, 3.526),
           0.01: (-7.743, -2.598, 6.748), 0.05: (-9.506, -2.610, 8.512)}
for r in core["breakdown"]["rows"]:
    w = bd_want.get(round(r["eps"], 4))
    if w:
        chk(f"5.2 eps={r['eps']} min rule", w[0], r["min_mean"], 6e-4)
        chk(f"5.2 eps={r['eps']} floored", w[1], r["capped_mean"], 6e-4)

# ---- section 7
solver_want = {0.0: (5.4e-4, 3.7e-5, 1.0e-4), 0.02: (1.1e-3, 3.8e-3, 1.0e-2),
               0.1: (6.4e-3, 5.6e-2, 1.5e-1)}
for r in sol["rows"]:
    w = solver_want.get(round(r["eps"], 4))
    if w:
        chk(f"7.1 eps={r['eps']} gap", w[0], r["gap"], max(1e-4, 0.15 * abs(w[0])))
        chk(f"7.1 eps={r['eps']} sd", w[1], r["sd_res"], max(1e-5, 0.15 * abs(w[1])))
        chk(f"7.1 eps={r['eps']} b_K sd", w[2], r["bK"] * r["sd_res"],
            max(1e-5, 0.15 * abs(w[2])))

print(f"{len(checks)} checks")
bad = [c for c in checks if not c[0]]
for ok, claim, pr, ac in bad:
    print(f"  MISMATCH  {claim:36s} printed={pr:>13.6g}  actual={ac:>13.6g}")
print()
print(f"passed {len(checks)-len(bad)}/{len(checks)}")
