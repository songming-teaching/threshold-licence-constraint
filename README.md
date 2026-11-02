# Threshold Selection Under a Non-Increasing Licence Constraint

Code and data for the paper *"The Threshold Parameter in Robust Losses: What the
Annealing Rule Is, What It Costs, and What It Buys."*

The min-annealing rule `theta_k = min(theta_{k-1}, T(r_k))` is used in five published
robust-recovery methods. This repository contains the analysis of what that rule
guarantees and what the guarantee costs.

## What is here

| directory | contents |
|---|---|
| `paper/` | the manuscript backbone (Markdown) |
| `code/verify/` | the theory checks: frontier, exchange rate, price, identities |
| `code/experiments/` | the solver experiments: capped-Frobenius PBCD, relaxations |
| `results/` | numbers behind every table, as JSON/CSV |
| `figures/` | three figures |

## The three results

1. **Characterisation.** The running minimum is the pointwise greatest element of the
   set of causal non-increasing schedules that never exceed the current estimate. It is
   *not* an isotonic regression of any order, and its L-infinity deviation is exactly
   twice that of the isotonic fit.

2. **Pricing.** For any non-increasing schedule, `bias + over-trust >= b_K`. The shifted
   running minimum attains the bound pathwise with an exchange rate of exactly one. The
   price has a closed quadrature:

   ```
   b_K = integral_0^inf [1 - F(t)^K - (1 - F(t))^K] dt
   ```

   which grows like `sqrt(2 ln K)` under Gaussian residuals and like `K^(1/nu)` under
   heavy tails.

3. **Scope.** On a real solver the rule is reproduced exactly, but the price is governed
   by the solver's effective history (1-3 iterations), not the iteration count (200).
   A controlled experiment shows that the threshold path does not determine which
   observations are retained, and that the retained set affects reconstruction
   independently of the frontier.

## Reproducing

Requires Python 3.10+, numpy, scipy, matplotlib, joblib, pandas.

```bash
export OPENBLAS_NUM_THREADS=1     # avoid oversubscription when using joblib

python code/verify/verify_core.py         # frontier, exchange, isotonic comparison
python code/verify/verify_final.py        # exact quadrature vs simulation
python code/verify/verify_regen.py        # price tables, expansion errors
python code/verify/verify_draft_numbers.py # audit: text figures vs stored results

python code/experiments/e1_frontier.py    # figure 1
python code/experiments/e2_price.py       # figure 2
python code/experiments/e3_solver_v2.py   # solver relaxation scan
python code/experiments/e5c_controlled.py # controlled keep-mask comparison
```

Total runtime for everything above is under fifteen minutes on a laptop CPU.

## A note on the solver

`code/experiments/pbcd.py` implements the proximal block-coordinate step of Li et al.,
"Robust Tensor Completion via Capped Frobenius Norm" (Algorithm 1, equations 28 and
38-41), reduced to the matrix case. The proximal term is retained: dropping it turns
the update into plain alternating least squares, which is a different algorithm and is
not a valid baseline.

Initialisation is from the SVD of the Winsorised observation matrix. An unwinsorised
SVD follows the outlier block rather than the signal, because the outlier block's
Frobenius norm exceeds the signal's; on the settings used here that alone moves the
reconstruction error from 0.035 to 0.81.

## Scope

The theory is about the *threshold sequence*. It does not claim that a schedule with
lower threshold error reconstructs better, and section 7.5 of the paper isolates the
reason: the retained-observation set is a further decision that the bias/over-trust
accounting does not price.

The theory also assumes independent, identically distributed threshold estimates. In a
converging solver that assumption fails, and the operative count is the transient
length. A law for the effective count is left open.
