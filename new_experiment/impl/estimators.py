import numpy as np


class HoldoutTransferMatrixEstimator:
    """Estimate transfer matrix P[j,k] = P(leaf_j | Y=k) using held-out data.

    Expects `leaf_indices` to be an array-like of leaf ids (node indices) for
    each validation instance and `y` to be the corresponding class labels.
    """

    def __init__(self, alpha=0.0):
        self.alpha = float(alpha)

    def fit(self, leaf_indices, y):
        leaf_indices = np.asarray(leaf_indices)
        y = np.asarray(y)
        self.leaves_ = np.unique(leaf_indices)
        self.classes_, inv = np.unique(y, return_inverse=True)
        n_leaves = len(self.leaves_)
        n_classes = len(self.classes_)
        P = np.zeros((n_leaves, n_classes), dtype=float)

        # map leaf id -> row index
        leaf_idx_map = {leaf: i for i, leaf in enumerate(self.leaves_)}

        for k, c in enumerate(self.classes_):
            mask = (y == c)
            denom = mask.sum()
            if denom == 0:
                if self.alpha > 0:
                    P[:, k] = np.ones(n_leaves, dtype=float) / float(n_leaves)
                continue
            # count occurrences per leaf
            rows = [leaf_idx_map[l] for l in leaf_indices[mask]]
            counts = np.bincount(rows, minlength=n_leaves)
            if self.alpha > 0:
                P[:, k] = (counts.astype(float) + self.alpha) / float(denom + self.alpha * n_leaves)
            else:
                P[:, k] = counts.astype(float) / float(denom)

        self.P_ = P
        self.n_leaves_, self.n_classes_ = P.shape
        self.leaf_to_row = {leaf: i for i, leaf in enumerate(self.leaves_)}
        return self

    def get_matrix(self):
        return self.P_
