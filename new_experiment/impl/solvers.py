import numpy as np


class EMQuantificationSolver:
    def __init__(self, tol=1e-6, max_iter=1000):
        self.tol = tol
        self.max_iter = max_iter

    def estimate_prevalence(self, counts, P, init_pi=None):
        counts = np.asarray(counts, dtype=float)
        N = counts.sum()
        n_leaves, n_classes = P.shape
        if N == 0:
            return np.ones(n_classes) / n_classes
        if init_pi is None:
            pi = np.ones(n_classes) / n_classes
        else:
            pi = np.asarray(init_pi, dtype=float)

        for it in range(self.max_iter):
            denom = P.dot(pi)  # shape (n_leaves,)
            denom[denom <= 0] = 1e-12
            # responsibilities r_jk = P[j,k] * pi_k / denom_j
            r = (P * pi[np.newaxis, :]) / denom[:, np.newaxis]
            expected = (counts[:, np.newaxis] * r).sum(axis=0)
            new_pi = expected / N
            if np.linalg.norm(new_pi - pi, ord=1) < self.tol:
                pi = new_pi
                break
            pi = new_pi
        # numerical corrections
        pi = np.maximum(pi, 0)
        s = pi.sum()
        if s == 0:
            return np.ones_like(pi) / len(pi)
        return pi / s


class ACCQuantificationSolver:
    """Constrained least-squares solver (approximate): solve min ||q - P pi||^2
    s.t. pi >= 0, sum(pi)=1. We solve an unconstrained LS then project to the simplex.
    """

    def estimate_prevalence(self, counts, P, init_pi=None):
        counts = np.asarray(counts, dtype=float)
        N = counts.sum()
        n_leaves, n_classes = P.shape
        if N == 0:
            return np.ones(n_classes) / n_classes
        q = counts / N

        # Solve least squares (regularized) for stability
        try:
            A = P.T.dot(P)
            b = P.T.dot(q)
            x = np.linalg.solve(A + 1e-8 * np.eye(A.shape[0]), b)
        except np.linalg.LinAlgError:
            x, *_ = np.linalg.lstsq(P, q, rcond=None)

        # project vector x to the probability simplex
        def proj_simplex(v):
            v = np.asarray(v)
            if v.size == 0:
                return v
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u)
            rho = np.nonzero(u + (1.0 - cssv) / (np.arange(1, len(v) + 1)) > 0)[0]
            if rho.size == 0:
                theta = 0.0
            else:
                rho = rho[-1]
                theta = (cssv[rho] - 1.0) / (rho + 1.0)
            w = np.maximum(v - theta, 0.0)
            return w

        pi = proj_simplex(np.maximum(x, 0.0))
        s = pi.sum()
        if s == 0:
            return np.ones_like(pi) / len(pi)
        return pi / s
