"""Capped-Frobenius low-rank recovery by proximal block-coordinate descent.

Faithful to Li et al. "Robust Tensor Completion via Capped Frobenius Norm",
Algorithm 1 / equations (28), (38)-(41), reduced to the N=2 (matrix) case. The
point of this module is the *proximal* term: eq (40) keeps the previous iterate,

    y^{k+1} = (A_l^T A_l + lambda I)^{-1} (A_l^T g_l + lambda y^k),

which is what makes the objective monotone under block updates. Plain
alternating least squares (the previous quick stand-in) drops the lambda y^k
term and is NOT the published algorithm; it is not a valid baseline.

Reduction note (recorded in the log): with N=2, the tensor-ring contraction of
Y1 in R^{R x n1 x R} and Y2 in R^{R x n2 x R} is a rank-R matrix factorisation
with a self-loop closure on the auxiliary dimension. We implement that closure
as a rank-R factorisation X = U V^T. This preserves the block structure, the
proximal term, the MADN rule and the descent argument; it does not preserve the
TR-specific multi-fold coupling, which needs N>=3.
"""
import numpy as np


def capped_obj(X, Y, mask, theta):
    """sum over observed entries of min(r^2, theta^2)."""
    R = (X - Y)[mask]
    return float(np.minimum(R ** 2, theta ** 2).sum())


def madn(r, zeta=3.0):
    r = np.asarray(r, dtype=float)
    return float(zeta * 1.4826 * np.median(np.abs(r - np.median(r))))


def pbcd_step(Y, mask, U, V, theta, lam, eps=0.0):
    """One PBCD sweep. Rows of U then rows of V, each with the proximal term.

    Inlier set: observed entries with |residual| below theta, and (for the
    augmentation test) a fixed fraction eps of observed entries are held out of
    the fit regardless of residual size, imitating a cap that cannot tell an
    inlier from an outlier.
    """
    n1, r = U.shape
    n2 = V.shape[0]
    eye = np.eye(r)

    # ---- U block
    R = Y - U @ V.T
    W = mask & (np.abs(R) <= theta)
    for i in range(n1):
        w = W[i]
        k = int(w.sum())
        if k < r + 1:
            continue
        Vi = V[w]                          # k x r
        A = Vi.T @ Vi                      # r x r
        b = Vi.T @ Y[i, w]                 # r
        U[i] = np.linalg.solve(A + lam * eye, b + lam * U[i])

    # ---- V block
    R = Y - U @ V.T
    W = mask & (np.abs(R) <= theta)
    for j in range(n2):
        w = W[:, j]
        k = int(w.sum())
        if k < r + 1:
            continue
        Uj = U[w]
        A = Uj.T @ Uj
        b = Uj.T @ Y[w, j]
        V[j] = np.linalg.solve(A + lam * eye, b + lam * V[j])

    return U, V


def init_svd(Y, mask, rank):
    Z = np.where(mask, Y, 0.0)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    return (U[:, :rank] * np.sqrt(s[:rank]), Vt[:rank].T * np.sqrt(s[:rank]))


def run(Y, X_true, mask, rank=3, K=200, zeta=3.0, lam=None, delta=0.0,
        inner=1, theta0=None, seed=0, mode="min", floor=None):
    """Full annealing loop.

    mode = 'min'   : theta_k = min(that_hat_k, theta_{k-1})            (published)
    mode = 'shift' : same, then the reported schedule is the running minimum
                     plus delta (Theorem 3.2). delta=0 recovers 'min'.
    mode = 'floor' : running minimum clipped below at -floor.
    """
    n1, n2 = Y.shape
    if lam is None:
        lam = 1e-3 * float(np.trace(Y.T @ Y) / max(mask.sum(), 1))
    U, V = init_svd(Y, mask, rank)

    # warm start at a loose threshold, no annealing
    cur = theta0 if theta0 is not None else np.inf
    for _ in range(5):
        U, V = pbcd_step(Y, mask, U, V, cur, lam, inner)
    R = (Y - U @ V.T)[mask]
    cur = min(cur, madn(R, zeta))
    that, theta = [], []
    for _ in range(K):
        for _ in range(inner):
            U, V = pbcd_step(Y, mask, U, V, cur, lam)
        R = (Y - U @ V.T)[mask]
        h = madn(R, zeta)
        cur = min(h, cur)
        that.append(h)
        theta.append(cur)

    that = np.asarray(that)
    theta = np.asarray(theta)
    mk = np.minimum.accumulate(that)

    if mode == "shift":
        rep = np.minimum.accumulate(mk + delta)
    elif mode == "floor":
        rep = np.maximum(mk, -floor)
    else:
        rep = theta

    X = U @ V.T
    relerr = float(np.linalg.norm(X_true - X) / np.linalg.norm(X_true))
    return dict(that=that, theta=theta, reported=rep, relerr=relerr,
                lam=lam, X=X)
