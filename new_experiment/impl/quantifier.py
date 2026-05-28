import numpy as np


class TreeBinQuantifier:
    """Wires a `QuantificationTree`, `TransferMatrixEstimator`, and `QuantificationSolver`.

    Usage:
        q = TreeBinQuantifier(tree, estimator, solver)
        q.fit(X_train, y_train, X_val, y_val)
        pi = q.quantify(X_test)
    """

    def __init__(self, tree, matrix_estimator, solver, min_calibration_samples=0):
        self.tree = tree
        self.matrix_estimator = matrix_estimator
        self.solver = solver
        self.min_calibration_samples = int(min_calibration_samples)

    def fit(self, X_train, y_train, X_val, y_val):
        self.tree.fit(X_train, y_train)
        leaf_indices_val = self.tree.get_leaf_indices(X_val)
        leaf_indices_val = np.asarray(leaf_indices_val)
        y_val = np.asarray(y_val)

        if self.min_calibration_samples > 0:
            counts_per_leaf = np.bincount(leaf_indices_val.astype(int))
            observed_leaves = np.unique(leaf_indices_val)
            keep_leaves = set(
                leaf for leaf in observed_leaves
                if counts_per_leaf[int(leaf)] >= self.min_calibration_samples
            )
            pruned_leaves = set(
                leaf for leaf in observed_leaves
                if leaf not in keep_leaves
            )
            self.pruned_leaves_ = sorted(pruned_leaves)

            # Build the transfer matrix using only well-supported leaves
            keep_mask = np.array([leaf in keep_leaves for leaf in leaf_indices_val], dtype=bool)
            leaf_indices_val_filtered = leaf_indices_val[keep_mask]
            y_val_filtered = y_val[keep_mask]

            self.matrix_estimator.fit(leaf_indices_val_filtered, y_val_filtered)
            self.P_ = self.matrix_estimator.get_matrix()
            self.leaves_ = self.matrix_estimator.leaves_
            self.leaf_to_row = self.matrix_estimator.leaf_to_row
            self.classes_ = self.matrix_estimator.classes_

            # Build a redirect map for pruned leaves:
            # Map each unsupported leaf to the nearest supported leaf based on
            # class-distribution similarity (using the raw counts from validation).
            # This ensures test instances landing in pruned leaves are redirected
            # rather than discarded.
            self.leaf_redirect_ = {}
            if len(pruned_leaves) > 0 and len(keep_leaves) > 0:
                # Compute class distribution profile for each observed leaf
                n_classes = len(self.classes_)
                class_to_idx = {c: i for i, c in enumerate(self.classes_)}

                # Profiles for supported leaves (keyed by leaf id)
                supported_profiles = {}
                for leaf in keep_leaves:
                    mask = leaf_indices_val == leaf
                    y_leaf = y_val[mask]
                    profile = np.zeros(n_classes, dtype=float)
                    for label in y_leaf:
                        idx = class_to_idx.get(label, None)
                        if idx is not None:
                            profile[idx] += 1
                    total = profile.sum()
                    if total > 0:
                        profile /= total
                    supported_profiles[leaf] = profile

                # For each pruned leaf, find the closest supported leaf
                supported_leaf_list = sorted(keep_leaves)
                supported_profile_matrix = np.array(
                    [supported_profiles[leaf] for leaf in supported_leaf_list]
                )

                for leaf in pruned_leaves:
                    mask = leaf_indices_val == leaf
                    y_leaf = y_val[mask]
                    profile = np.zeros(n_classes, dtype=float)
                    for label in y_leaf:
                        idx = class_to_idx.get(label, None)
                        if idx is not None:
                            profile[idx] += 1
                    total = profile.sum()
                    if total > 0:
                        profile /= total

                    # Find closest supported leaf by L2 distance of class profiles
                    dists = np.linalg.norm(
                        supported_profile_matrix - profile[np.newaxis, :], axis=1
                    )
                    closest_idx = int(np.argmin(dists))
                    closest_leaf = supported_leaf_list[closest_idx]
                    self.leaf_redirect_[leaf] = closest_leaf
        else:
            self.pruned_leaves_ = []
            self.leaf_redirect_ = {}
            self.matrix_estimator.fit(leaf_indices_val, y_val)
            self.P_ = self.matrix_estimator.get_matrix()
            self.leaves_ = self.matrix_estimator.leaves_
            self.leaf_to_row = self.matrix_estimator.leaf_to_row
            self.classes_ = self.matrix_estimator.classes_

        return self

    def quantify(self, X_test, init_pi=None):
        leaf_indices_test = self.tree.get_leaf_indices(X_test)

        # Map test leaf ids to estimator rows.
        # Redirect pruned leaves to their nearest supported neighbor
        # instead of discarding them.
        rows = np.empty(len(leaf_indices_test), dtype=int)
        for i, leaf in enumerate(leaf_indices_test):
            # First check if leaf is directly in the transfer matrix
            row = self.leaf_to_row.get(leaf, -1)
            if row >= 0:
                rows[i] = row
            else:
                # Try redirecting via the redirect map
                redirected_leaf = self.leaf_redirect_.get(leaf, None)
                if redirected_leaf is not None:
                    rows[i] = self.leaf_to_row.get(redirected_leaf, -1)
                else:
                    rows[i] = -1

        mask = rows >= 0
        counts = np.bincount(rows[mask], minlength=self.P_.shape[0])

        # Call solver
        try:
            pi = self.solver.estimate_prevalence(counts, self.P_, init_pi=init_pi)
        except TypeError:
            pi = self.solver.estimate_prevalence(counts, self.P_)
        return pi