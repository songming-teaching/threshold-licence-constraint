# Threshold Selection Under a Non-Increasing Licence Constraint

Code for the min-annealing threshold rule used in robust low-rank recovery:

```
theta_k = min(theta_{k-1}, T(r_k))
```

Five published methods use this rule in exactly this form — hybrid ordinary-Welsch
(HOW), hybrid quadratic-Frobenius (HQF), `l_{p,epsilon}` tensor completion,
capped-Frobenius tensor completion, and robust elliptic localisation — with a
different statistic `T` each time but the same update. This repository contains the
solver, the numerical experiments, and the numbers behind every figure.

## What is here

| directory | contents |
|---|---|
| `code/experiments/` | capped-Frobenius solver and the experiment scripts |
| `code/verify/` | numerical checks of the identities the analysis rests on |
| `results/` | the numbers behind every figure, as JSON/CSV |
| `figures/` | three figures |

## Quick start

Requires Python 3.10+, numpy, scipy, matplotlib, joblib, pandas.

```bash
export OPENBLAS_NUM_THREADS=1     # avoid thread oversubscription under joblib

python code/experiments/e1_frontier.py    # figure 1: the frontier
python code/experiments/e2_price.py       # figure 2: the price
python code/experiments/e3_solver_v2.py   # solver relaxation scan
python code/experiments/e5c_controlled.py # controlled keep-mask comparison
```

Everything runs in under fifteen minutes on a laptop CPU. No GPU is used, and none
is needed: the per-iteration cost is `O(n r^2)` with `n` around 100 and `r` around 3,
so the arithmetic is smaller than the interpreter overhead around it.

## The solver

`code/experiments/pbcd.py` implements the proximal block-coordinate step of

> X.-P. Li et al., "Robust Tensor Completion via Capped Frobenius Norm,"
> IEEE Trans. Neural Netw. Learn. Syst., 2024.

Algorithm 1, equations (28) and (38)-(41), reduced to the matrix case. Two things
about it are worth knowing before you use it.

**The proximal term matters.** Equation (40) keeps the previous iterate,

```
y^{k+1} = (A_l^T A_l + lambda I)^{-1} (A_l^T g_l + lambda y^k),
```

and dropping the `lambda y^k` turns the update into plain alternating least squares.
That is a different algorithm, it does not have the descent property, and it is not a
valid baseline against the published method. `pbcd.py` keeps the term.

**Initialisation matters more than you would expect.** The natural choice, the rank-`r`
SVD of the masked observation matrix, fails badly once the observations are corrupted:
the outlier block's Frobenius norm exceeds the signal's, so the leading singular
vectors follow the outliers. On the settings used here that alone moves the
reconstruction error from 0.035 to 0.81 — from working to useless. The fix is to
Winsorise the observations at `3 x MADN` before the decomposition, which is what
`init_winsor` does. Every experiment in this repository uses it.

## Data conventions

Simulation columns are cross-checks against quadrature, never the other way round:
where the two disagree beyond two standard errors, the simulation is presumed at fault
until the quadrature is shown otherwise. Monte-Carlo replication counts grow with the
problem size rather than shrinking, because the variance of an extreme order statistic
increases with the number of draws.

Seeds: `20261102` for the solver experiments, `112358` for the exchange table,
`60601` for the L-infinity identity.

## Scope

The analysis concerns the threshold sequence. It does not claim that a schedule with
lower threshold error reconstructs better — the retained-observation set is a further
decision, and the experiments here show it moves the reconstruction independently of
any threshold quantity.

The analysis also assumes independently distributed threshold estimates. In a
converging solver that assumption fails: the estimate freezes after a few iterations,
so the operative count is the transient length rather than the iteration count. A law
for that effective count is not given here. The scripts record the full threshold path
per iteration so that the question can be worked on directly.

## Citation

Please cite the paper once it is available. Until then, cite this repository.
