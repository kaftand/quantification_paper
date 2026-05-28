import numpy as np
from collections import Counter
from .tree import ClassificationTree
from .estimators import HoldoutTransferMatrixEstimator
from .solvers import EMQuantificationSolver
from copy import deepcopy
import os
import subprocess
import tempfile
import json
from .weka_tree import parse_j48_tree, predict_leaves


def _majority_label(y):
    if len(y) == 0:
        return None
    return Counter(y).most_common(1)[0][0]


def _build_qe_tree_optimized(X, y, max_depth, min_samples_leaf, squared=False,
                              max_features=None, max_thresholds=None, rng=None):
    """Optimized QE tree building with vectorized split evaluation."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    n_classes = len(classes)
    n_samples, n_features = X.shape

    # Pre-encode y as integer class indices (avoids repeated searches)
    class_to_index = {label: i for i, label in enumerate(classes)}
    y_encoded = np.array([class_to_index[v] for v in y], dtype=np.int32)

    # Pre-sort each feature column and store sorted indices
    # This allows us to efficiently compute split counts
    sorted_indices = np.empty((n_features, n_samples), dtype=np.intp)
    sorted_X = np.empty((n_features, n_samples), dtype=np.float64)
    for feat in range(n_features):
        order = np.argsort(X[:, feat], kind='mergesort')
        sorted_indices[feat] = order
        sorted_X[feat] = X[order, feat]

    def compute_qe_norm(class_counts, total, squared_mode):
        """Compute QE norm from class counts array."""
        if total == 0:
            return 0.0
        maj_idx = np.argmax(class_counts)
        maj_count = class_counts[maj_idx]

        if squared_mode:
            # E[maj] = abs(FP^2 - FN^2) = abs((total-maj)^2 - 0) = (total-maj)^2
            # E[other] = abs(0 - count[c]^2) = count[c]^2
            E = class_counts.astype(np.float64) ** 2  # FN^2 for non-majority
            E[maj_idx] = (total - maj_count) ** 2  # FP^2 for majority
        else:
            # E[maj] = abs(FP - FN) = total - maj_count
            # E[other] = abs(0 - count[c]) = count[c]
            E = class_counts.astype(np.float64)
            E[maj_idx] = float(total - maj_count)

        return np.sqrt(np.dot(E, E))

    def build(indices, depth):
        """Build tree node. `indices` is an array of sample indices in this node."""
        n_node = len(indices)
        y_node = y_encoded[indices]

        # Compute class counts for this node
        class_counts = np.bincount(y_node, minlength=n_classes)
        maj_idx = int(np.argmax(class_counts))
        prediction = classes[maj_idx]

        node = {
            'n': n_node,
            'prediction': prediction,
            'is_leaf': True,
            'left': None,
            'right': None,
        }

        if (max_depth is not None and depth >= max_depth) or n_node <= min_samples_leaf:
            return node

        parent_obj = compute_qe_norm(class_counts, n_node, squared)
        if parent_obj <= 0:
            return node

        # Create a set for O(1) membership testing
        node_set = set(indices)

        # Determine which features to evaluate
        if max_features is None:
            feature_indices = range(n_features)
        else:
            nonlocal rng
            if rng is None:
                rng = np.random.RandomState(0)
            if 0 < max_features <= 1:
                count = max(1, int(round(max_features * n_features)))
            else:
                count = int(max_features)
            count = max(1, min(count, n_features))
            feature_indices = rng.choice(n_features, size=count, replace=False)

        best_delta = 0.0
        best_feat = -1
        best_threshold = None
        best_split_pos = -1  # position in sorted order for the best split

        for feat in feature_indices:
            # Get the sorted order for this feature, filtered to node indices
            # We walk through the globally sorted order and pick out node members
            feat_sorted_idx = sorted_indices[feat]
            feat_sorted_vals = sorted_X[feat]

            # Filter to indices in this node, maintaining sort order
            # For small nodes relative to dataset, iterate over node indices and sort
            # For large nodes, iterate over global sorted order
            if n_node < n_samples * 0.5:
                # Sort node indices by feature value
                node_feat_vals = X[indices, feat]
                local_order = np.argsort(node_feat_vals, kind='mergesort')
                local_sorted_indices = indices[local_order]
                local_sorted_vals = node_feat_vals[local_order]
            else:
                # Walk global sorted order, filter to node members
                mask_in_node = np.array([idx in node_set for idx in feat_sorted_idx], dtype=bool)
                local_sorted_indices = feat_sorted_idx[mask_in_node]
                local_sorted_vals = feat_sorted_vals[mask_in_node]

            # Now sweep through sorted values computing left/right class counts
            if len(local_sorted_vals) <= 1:
                continue

            # Check if all values are the same
            if local_sorted_vals[0] == local_sorted_vals[-1]:
                continue

            # Initialize left counts as empty, right counts as full node
            left_counts = np.zeros(n_classes, dtype=np.int64)
            right_counts = class_counts.copy().astype(np.int64)

            # Determine threshold candidates
            # We sweep left-to-right, and at each boundary where value changes,
            # we evaluate the split
            n_local = len(local_sorted_indices)

            # Apply max_thresholds subsampling
            if max_thresholds is not None:
                # Find all unique split positions
                split_positions = []
                for i in range(n_local - 1):
                    if local_sorted_vals[i] < local_sorted_vals[i + 1]:
                        split_positions.append(i)
                if len(split_positions) == 0:
                    continue
                if len(split_positions) > max_thresholds:
                    step = len(split_positions) / max_thresholds
                    selected = [split_positions[int(j * step)] for j in range(max_thresholds)]
                    split_positions = selected
                # Evaluate only at selected positions
                # We need cumulative class counts up to each position
                # Build cumulative counts
                y_local = y_encoded[local_sorted_indices]
                cum_counts = np.zeros((n_local, n_classes), dtype=np.int64)
                cum_counts[0, y_local[0]] = 1
                for i in range(1, n_local):
                    cum_counts[i] = cum_counts[i - 1]
                    cum_counts[i, y_local[i]] += 1

                for pos in split_positions:
                    left_n = pos + 1
                    right_n = n_local - left_n
                    if left_n < min_samples_leaf or right_n < min_samples_leaf:
                        continue
                    lc = cum_counts[pos]
                    rc = class_counts - lc
                    threshold = (local_sorted_vals[pos] + local_sorted_vals[pos + 1]) / 2.0

                    left_obj = compute_qe_norm(lc, left_n, squared)
                    right_obj = compute_qe_norm(rc, right_n, squared)
                    child_obj = (left_n * left_obj + right_n * right_obj) / n_node
                    delta = parent_obj - child_obj

                    if delta > best_delta:
                        best_delta = delta
                        best_feat = feat
                        best_threshold = threshold
            else:
                # Sweep all valid split points
                y_local = y_encoded[local_sorted_indices]

                for i in range(n_local - 1):
                    left_counts[y_local[i]] += 1
                    right_counts[y_local[i]] -= 1

                    # Only evaluate when value changes (valid split point)
                    if local_sorted_vals[i] >= local_sorted_vals[i + 1]:
                        continue

                    left_n = i + 1
                    right_n = n_local - left_n
                    if left_n < min_samples_leaf or right_n < min_samples_leaf:
                        continue

                    threshold = (local_sorted_vals[i] + local_sorted_vals[i + 1]) / 2.0

                    left_obj = compute_qe_norm(left_counts, left_n, squared)
                    right_obj = compute_qe_norm(right_counts, right_n, squared)
                    child_obj = (left_n * left_obj + right_n * right_obj) / n_node
                    delta = parent_obj - child_obj

                    if delta > best_delta:
                        best_delta = delta
                        best_feat = feat
                        best_threshold = threshold

        if best_delta <= 0 or best_feat < 0:
            return node

        # Apply the best split
        left_mask = X[indices, best_feat] <= best_threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        node['is_leaf'] = False
        node['feature'] = int(best_feat)
        node['threshold'] = best_threshold
        node['left'] = build(left_indices, depth + 1)
        node['right'] = build(right_indices, depth + 1)
        return node

    all_indices = np.arange(n_samples, dtype=np.intp)
    tree = build(all_indices, 0)
    return tree, classes


def _count_leaves(node):
    if node['is_leaf']:
        return 1
    return _count_leaves(node['left']) + _count_leaves(node['right'])


def _assign_leaf_ids(node, next_id=0):
    if node['is_leaf']:
        node['_leaf_id'] = next_id
        return next_id + 1
    next_id = _assign_leaf_ids(node['left'], next_id)
    next_id = _assign_leaf_ids(node['right'], next_id)
    return next_id


def _apply_leaf_id(node, x):
    if node['is_leaf']:
        return node['_leaf_id']
    if x[node['feature']] <= node['threshold']:
        return _apply_leaf_id(node['left'], x)
    return _apply_leaf_id(node['right'], x)


def _get_leaf_indices_batch(tree, X):
    """Vectorized leaf index assignment for all samples."""
    n = X.shape[0]
    result = np.empty(n, dtype=np.int32)
    # Use iterative traversal with stack to avoid Python recursion overhead per sample
    # Process all samples together through the tree
    _assign_leaves_recursive(tree, np.arange(n, dtype=np.intp), X, result)
    return result


def _assign_leaves_recursive(node, sample_indices, X, result):
    """Recursively assign leaf IDs to batches of samples."""
    if node['is_leaf']:
        result[sample_indices] = node['_leaf_id']
        return
    if len(sample_indices) == 0:
        return
    feat = node['feature']
    threshold = node['threshold']
    vals = X[sample_indices, feat]
    left_mask = vals <= threshold
    left_samples = sample_indices[left_mask]
    right_samples = sample_indices[~left_mask]
    _assign_leaves_recursive(node['left'], left_samples, X, result)
    _assign_leaves_recursive(node['right'], right_samples, X, result)


class QuantificationErrorBalancingTree:
    """Optimized greedy QEB tree."""
    def __init__(self, max_depth=3, min_samples_leaf=1, max_features=None,
                 max_thresholds=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.random_state = random_state

    def fit(self, X, y):
        rng = None if self.random_state is None else np.random.RandomState(self.random_state)
        self.tree_, self.classes_ = _build_qe_tree_optimized(
            X, y,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            squared=False,
            max_features=self.max_features,
            max_thresholds=self.max_thresholds,
            rng=rng,
        )
        self.n_leaves_ = _count_leaves(self.tree_)
        _assign_leaf_ids(self.tree_)
        return self

    def get_leaf_indices(self, X):
        X = np.asarray(X, dtype=np.float64)
        return _get_leaf_indices_batch(self.tree_, X)


class ClassificationQuantificationBalancingTree(QuantificationErrorBalancingTree):
    """Optimized greedy QCQB tree."""
    def __init__(self, max_depth=3, min_samples_leaf=1, max_features=None,
                 max_thresholds=None, random_state=None, jar_path=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.random_state = random_state
        self.jar_path = jar_path

    def fit(self, X, y):
        rng = None if self.random_state is None else np.random.RandomState(self.random_state)
        self.tree_, self.classes_ = _build_qe_tree_optimized(
            X, y,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            squared=True,
            max_features=self.max_features,
            max_thresholds=self.max_thresholds,
            rng=rng,
        )
        self.n_leaves_ = _count_leaves(self.tree_)
        _assign_leaf_ids(self.tree_)
        return self


class ConditionNumberTree:
    """Greedy tree that selects splits minimizing condition number."""
    def __init__(self, max_depth=1, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X_val is None or y_val is None:
            X_val = X
            y_val = y
        X_val = np.asarray(X_val, dtype=np.float64)
        y_val = np.asarray(y_val)

        def build(mask, depth):
            node = {}
            y_node = y[mask]
            node['n'] = mask.sum()
            node['prediction'] = _majority_label(y_node)
            node['is_leaf'] = True
            node['left'] = None
            node['right'] = None
            if depth >= self.max_depth or mask.sum() <= self.min_samples_leaf:
                return node
            best = None
            parent_leaves = [mask]
            parent_P = _compute_transfer_matrix(parent_leaves, X_val, y_val)
            parent_cond = _cond(parent_P)
            for feat in range(X.shape[1]):
                vals = np.unique(X[mask, feat])
                if vals.size <= 1:
                    continue
                thresholds = (vals[:-1] + vals[1:]) / 2.0
                for t in thresholds:
                    left_mask = mask & (X[:, feat] <= t)
                    right_mask = mask & (X[:, feat] > t)
                    if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                        continue
                    leaves = [left_mask, right_mask]
                    P = _compute_transfer_matrix(leaves, X_val, y_val)
                    cond = _cond(P)
                    if best is None or cond < best[0]:
                        best = (cond, feat, t, left_mask, right_mask)
            if best is None:
                return node
            _, feat, t, left_mask, right_mask = best
            node['is_leaf'] = False
            node['feature'] = feat
            node['threshold'] = t
            node['left'] = build(left_mask, depth + 1)
            node['right'] = build(right_mask, depth + 1)
            return node

        root_mask = np.ones(X.shape[0], dtype=bool)
        self.tree_ = build(root_mask, 0)
        self.X_ = X
        self.y_ = y
        self.n_leaves_ = self._count_leaves(self.tree_)
        self.classes_ = np.unique(y)
        return self

    def _count_leaves(self, node):
        if node['is_leaf']:
            return 1
        return self._count_leaves(node['left']) + self._count_leaves(node['right'])

    def _apply_node(self, node, x):
        if node['is_leaf']:
            return id(node)
        if x[node['feature']] <= node['threshold']:
            return self._apply_node(node['left'], x)
        else:
            return self._apply_node(node['right'], x)

    def get_leaf_indices(self, X):
        X = np.asarray(X)
        ids = [self._apply_node(self.tree_, X[i]) for i in range(X.shape[0])]
        uniq = {}
        out = np.zeros(len(ids), dtype=int)
        for i, v in enumerate(ids):
            if v not in uniq:
                uniq[v] = len(uniq)
            out[i] = uniq[v]
        return out


def _compute_transfer_matrix(leaves_masks, X_val, y_val):
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)
    n_leaves = len(leaves_masks)
    classes = np.unique(y_val)
    P = np.zeros((n_leaves, classes.size), dtype=float)

    if leaves_masks and len(leaves_masks[0]) == len(X_val):
        for j, lm in enumerate(leaves_masks):
            for k, c in enumerate(classes):
                denom = (y_val == c).sum()
                if denom == 0:
                    P[j, k] = 0.0
                else:
                    P[j, k] = ((lm) & (y_val == c)).sum() / float(denom)
        return P

    for i in range(X_val.shape[0]):
        j = i % n_leaves
        k = np.where(classes == y_val[i])[0][0]
        P[j, k] += 1
    for idx, c in enumerate(classes):
        col_sum = P[:, idx].sum()
        if col_sum > 0:
            P[:, idx] /= col_sum
    return P


def _cond(M):
    try:
        s = np.linalg.svd(M, compute_uv=False)
        if s.size == 0:
            return np.inf
        if np.min(s) <= 0:
            return np.inf
        return float(np.max(s) / np.min(s))
    except Exception:
        return np.inf


class RandomForestTree:
    """Random forest of ClassificationTree wrappers."""
    def __init__(self, n_estimators=50, max_depth=3, min_samples_leaf=1,
                 max_features='sqrt', mode='per_tree', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.mode = mode
        self.random_state = random_state
        self.trees = []
        self.matrix_estimators = []

    def fit(self, X_train, y_train, X_val, y_val):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y_train)
        for i in range(self.n_estimators):
            idx = rng.randint(0, X_train.shape[0], size=X_train.shape[0])
            Xb = X_train[idx]
            yb = y_train[idx]
            tree = ClassificationTree(max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf,
                                      random_state=rng.randint(0, 2**31 - 1))
            tree.fit(Xb, yb)
            est = HoldoutTransferMatrixEstimator()
            leaf_indices_val = tree.get_leaf_indices(X_val)
            est.fit(leaf_indices_val, y_val)
            self.trees.append(tree)
            self.matrix_estimators.append(est)
        return self

    def quantify(self, X_test, solver=None):
        if solver is None:
            solver = EMQuantificationSolver()
        X_test = np.asarray(X_test)
        if self.mode == 'per_tree':
            pis = []
            for tree, est in zip(self.trees, self.matrix_estimators):
                rows = np.array([est.leaf_to_row.get(l, -1)
                                 for l in tree.get_leaf_indices(X_test)])
                mask = rows >= 0
                counts = np.bincount(rows[mask], minlength=est.P_.shape[0])
                try:
                    pi = solver.estimate_prevalence(counts, est.P_)
                except TypeError:
                    pi = solver.estimate_prevalence(counts, est.P_, None)
                pis.append(pi)
            return np.mean(np.vstack(pis), axis=0)
        else:
            P_list = []
            counts_list = []
            for tree, est in zip(self.trees, self.matrix_estimators):
                rows = np.array([est.leaf_to_row.get(l, -1)
                                 for l in tree.get_leaf_indices(X_test)])
                mask = rows >= 0
                counts = np.bincount(rows[mask], minlength=est.P_.shape[0])
                P_list.append(est.P_)
                counts_list.append(counts)
            P_big = np.vstack(P_list)
            counts_big = np.concatenate(counts_list)
            try:
                pi = solver.estimate_prevalence(counts_big, P_big)
            except TypeError:
                pi = solver.estimate_prevalence(counts_big, P_big, None)
            return pi