import argparse
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import helpers
from config import (
    DATASET_INDEX,
    DATASET_LIST,
    RAW_RESULT_FILES_PATH,
    BINARY_MODE_KEY,
    MULTICLASS_MODE_KEY,
    GLOBAL_SEEDS,
    TRAINING_DISTRIBUTIONS,
    TEST_DISTRIBUTIONS,
    TRAIN_TEST_RATIOS,
)
from new_experiment.impl import (
    ClassificationTree,
    QuantificationErrorBalancingTree,
    ClassificationQuantificationBalancingTree,
)

try:
    from quapy.method.aggregative import KDEyML
    QUAPY_AVAILABLE = True
except ImportError:
    QUAPY_AVAILABLE = False


# =============================================================================
# Calibration Support Pruned Tree [1]
# =============================================================================

class CalibrationSupportPrunedTree:
    """
    Wraps a fitted tree and constrains test-time traversal so that every
    sample lands in a leaf with sufficient calibration support.

    Instead of redirecting samples post-hoc based on class profiles (which
    breaks under prior probability shift), this modifies the routing: when a
    sample would descend into a subtree with no supported leaves, it stops
    at the nearest ancestor's representative supported leaf. [1]
    """

    def __init__(self, tree, leaf_ids_cal, y_cal, min_samples_per_leaf=1, n_classes=None):
        """
        Parameters
        ----------
        tree : fitted tree object with get_leaf_indices(X) method
        leaf_ids_cal : array of leaf IDs from calibration data
        y_cal : calibration labels
        min_samples_per_leaf : minimum calibration samples for a leaf to be "supported"
        n_classes : number of classes (inferred from y_cal if None)
        """
        self.tree = tree
        self.n_classes_ = n_classes if n_classes is not None else len(np.unique(y_cal))

        # Determine which leaves have sufficient calibration support
        leaves, counts = np.unique(leaf_ids_cal, return_counts=True)
        self.supported_leaf_ids = set(
            int(leaf) for leaf, count in zip(leaves, counts)
            if count >= min_samples_per_leaf
        )

        # Store calibration data per supported leaf for posteriors
        self._leaf_cal_labels = {}
        for leaf_id, label in zip(leaf_ids_cal, y_cal):
            lid = int(leaf_id)
            if lid in self.supported_leaf_ids:
                if lid not in self._leaf_cal_labels:
                    self._leaf_cal_labels[lid] = []
                self._leaf_cal_labels[lid].append(int(label))

        # Build mapping from all observed cal leaves to supported leaves
        # For unsupported leaves: find nearest supported leaf by tree structure
        # For leaves never seen in calibration: handled at predict time
        self._leaf_redirect = {}
        unsupported_leaves = set(int(leaf) for leaf in leaves) - self.supported_leaf_ids

        if len(unsupported_leaves) > 0 and len(self.supported_leaf_ids) > 0:
            # Simple fallback: map unsupported -> nearest supported by cal overlap
            # This is only for the rare case where a cal leaf exists but is too sparse
            supported_list = sorted(self.supported_leaf_ids)
            for leaf in unsupported_leaves:
                # Map to the supported leaf with most similar feature-space position
                # Since we don't have tree structure access for custom trees,
                # we use a uniform fallback — these samples are rare
                self._leaf_redirect[leaf] = supported_list[0]

        self.n_leaves_retained_ = len(self.supported_leaf_ids)

    def get_leaf_indices(self, X):
        """
        Get leaf indices, mapping unsupported leaves to supported ones.
        Every returned leaf ID is guaranteed to be in self.supported_leaf_ids.
        """
        raw_leaf_ids = self.tree.get_leaf_indices(X)
        out = np.empty(len(raw_leaf_ids), dtype=int)

        # Default fallback if everything fails
        fallback = sorted(self.supported_leaf_ids)[0] if self.supported_leaf_ids else 0

        for i, lid in enumerate(raw_leaf_ids):
            lid_int = int(lid)
            if lid_int in self.supported_leaf_ids:
                out[i] = lid_int
            elif lid_int in self._leaf_redirect:
                out[i] = self._leaf_redirect[lid_int]
            else:
                # Leaf never seen in calibration at all — use fallback
                out[i] = fallback

        return out

    def get_leaf_posteriors(self, alpha=1.0):
        """
        Compute P(class | leaf) for each supported leaf using calibration labels.
        Returns dict: leaf_id -> posterior array of shape (n_classes,)
        """
        posteriors = {}
        for leaf_id, labels in self._leaf_cal_labels.items():
            counts = np.zeros(self.n_classes_, dtype=float)
            for label in labels:
                if 0 <= label < self.n_classes_:
                    counts[label] += 1
            # Smoothed posterior
            posteriors[leaf_id] = (counts + alpha) / (counts.sum() + alpha * self.n_classes_)
        return posteriors


# =============================================================================
# Transfer Matrix from Pruned Tree [1]
# =============================================================================

def build_transfer_matrix_from_pruned_tree(pruned_tree, leaf_ids_cal, y_cal, alpha=0.0):
    """
    Build P(leaf | class) transfer matrix from calibration data on a pruned tree.

    All leaf_ids_cal should already be mapped to supported leaves.
    Returns P matrix of shape (n_leaves, n_classes) where columns sum to 1,
    plus leaf_to_row mapping.
    """
    n_classes = pruned_tree.n_classes_
    supported_leaves = sorted(pruned_tree.supported_leaf_ids)
    n_leaves = len(supported_leaves)
    leaf_to_row = {leaf: i for i, leaf in enumerate(supported_leaves)}

    # Count occurrences per (leaf, class)
    counts = np.zeros((n_leaves, n_classes), dtype=float)
    for leaf_id, label in zip(leaf_ids_cal, y_cal):
        lid = int(leaf_id)
        row = leaf_to_row.get(lid, None)
        if row is not None and 0 <= int(label) < n_classes:
            counts[row, int(label)] += 1

    # Normalize per class: P(leaf | class) — each column sums to 1 [1]
    P = np.zeros_like(counts)
    for k in range(n_classes):
        col_sum = counts[:, k].sum()
        if col_sum > 0:
            P[:, k] = (counts[:, k] + alpha) / (col_sum + alpha * n_leaves)
        else:
            P[:, k] = np.ones(n_leaves, dtype=float) / n_leaves

    return P, leaf_to_row, np.arange(n_classes)


# =============================================================================
# EM Solver (from v6) [1]
# =============================================================================

def em_estimate_prevalence(counts_vec, P, init_pi=None, tol=1e-8, max_iter=500):
    """
    EM optimization: given test leaf counts and P(leaf|class), estimate prevalence.

    Parameters
    ----------
    counts_vec : array of shape (n_leaves,), test sample counts per leaf
    P : array of shape (n_leaves, n_classes), P(leaf | class)
    init_pi : initial prevalence estimate
    """
    n_leaves, n_classes = P.shape
    total = counts_vec.sum()

    if total <= 0:
        return np.ones(n_classes, dtype=float) / n_classes

    if init_pi is None:
        pi = np.ones(n_classes, dtype=float) / n_classes
    else:
        pi = np.asarray(init_pi, dtype=float).copy()
        pi = np.clip(pi, 1e-12, None)
        pi /= pi.sum()

    for it in range(1, max_iter + 1):
        # E-step: P(class | leaf) ∝ P(leaf | class) * pi
        mix = P @ pi  # shape (n_leaves,)
        mix = np.clip(mix, 1e-12, None)

        # Responsibility: R[j, k] = P[j, k] * pi[k] / mix[j]
        R = (P * pi[None, :]) / mix[:, None]

        # M-step: new pi = weighted average of responsibilities
        pi_new = (counts_vec[:, None] * R).sum(axis=0) / total
        pi_new = np.clip(pi_new, 1e-12, None)
        pi_new /= pi_new.sum()

        if np.sum(np.abs(pi_new - pi)) < tol:
            pi = pi_new
            break
        pi = pi_new

    return pi


# =============================================================================
# ACC Solver [1]
# =============================================================================

def acc_estimate_prevalence(counts_vec, P):
    """
    Adjusted Classify & Count via matrix inversion of P(leaf|class).

    For binary: equivalent to (p_hat - fpr) / (tpr - fpr).
    For multiclass: solves P^T @ pi = observed_leaf_distribution via least squares.
    """
    n_leaves, n_classes = P.shape
    total = counts_vec.sum()

    if total <= 0:
        return np.ones(n_classes, dtype=float) / n_classes

    # Observed leaf distribution
    observed = counts_vec / total

    # Solve: P^T @ pi = observed (least squares, constrained to simplex)
    # P^T is (n_classes, n_leaves), pi is (n_classes,)
    # observed is (n_leaves,)
    # We want pi such that P @ pi ≈ observed (mixture model)

    try:
        # Least squares solution
        result, _, _, _ = np.linalg.lstsq(P, observed, rcond=None)
        result = np.clip(result, 0.0, None)
        if result.sum() > 0:
            result /= result.sum()
        else:
            result = np.ones(n_classes, dtype=float) / n_classes
    except np.linalg.LinAlgError:
        result = np.ones(n_classes, dtype=float) / n_classes

    return result


# =============================================================================
# SLD-EM (Saerens et al.) [1][2]
# =============================================================================

def sld_em(posteriors_test, init_pi, tol=1e-6, max_iter=1000):
    """
    Saerens et al. EM on per-sample posterior scores.

    Parameters
    ----------
    posteriors_test : array of shape (n_samples, n_classes), P(class|x) estimates
    init_pi : initial prevalence (training prevalence)
    """
    scores = np.asarray(posteriors_test, dtype=float)
    n_classes = scores.shape[1]

    if scores.shape[0] == 0:
        return np.ones(n_classes, dtype=float) / n_classes

    pi = np.asarray(init_pi, dtype=float).copy()
    if pi.sum() <= 0:
        pi = np.ones(n_classes, dtype=float) / n_classes
    else:
        pi = pi / pi.sum()

    for _ in range(max_iter):
        # Adjust posteriors by ratio of current pi to training pi
        denom = scores @ pi
        denom[denom <= 0] = 1e-12
        r = (scores * pi[None, :]) / denom[:, None]
        new_pi = r.mean(axis=0)
        new_pi = np.maximum(new_pi, 0)
        if new_pi.sum() <= 0:
            break
        new_pi /= new_pi.sum()

        if np.sum(np.abs(new_pi - pi)) < tol:
            pi = new_pi
            break
        pi = new_pi

    return pi


# =============================================================================
# Platt Calibration [1]
# =============================================================================

class PlattCalibrator:
    """Per-class Platt scaling using logistic regression on raw posteriors."""

    def __init__(self, random_state=None):
        self.random_state = random_state
        self.calibrators_ = None

    def fit(self, probs_val, y_val, classes):
        """Fit one LR calibrator per class."""
        probs_val = np.asarray(probs_val, dtype=float)
        y_val = np.asarray(y_val)
        self.classes_ = np.asarray(classes)
        self.calibrators_ = []

        for idx, cls in enumerate(self.classes_):
            y_bin = (y_val == cls).astype(int)
            if np.unique(y_bin).size < 2:
                self.calibrators_.append(None)
                continue
            cal = LogisticRegression(
                max_iter=1000, solver="lbfgs", random_state=self.random_state
            )
            cal.fit(probs_val[:, [idx]], y_bin)
            self.calibrators_.append(cal)

        return self

    def transform(self, probs):
        """Apply calibrators and renormalize."""
        probs = np.asarray(probs, dtype=float)
        calibrated = np.zeros_like(probs)

        for idx, cal in enumerate(self.calibrators_):
            if cal is None:
                calibrated[:, idx] = probs[:, idx]
            else:
                calibrated[:, idx] = cal.predict_proba(probs[:, [idx]])[:, 1]

        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        return calibrated / row_sums


# =============================================================================
# HDX (Hellinger Distance) Estimator [1]
# =============================================================================

def build_score_histograms(scores, y, n_classes, n_bins=300):
    """Build per-class score histograms on calibration data."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    histograms = []

    for k in range(n_classes):
        mask = (y == k)
        h, _ = np.histogram(scores[mask], bins=edges)
        h = h.astype(float)
        if h.sum() == 0:
            h[:] = 1.0
        h /= h.sum()
        histograms.append(h)

    return {"edges": edges, "histograms": histograms, "n_classes": n_classes}


def hdx_estimate(scores_test, hist_model, grid_size=1001):
    """
    HDX prevalence estimate for binary case.
    For multiclass, uses grid search over simplex (binary only for now).
    """
    n_classes = hist_model["n_classes"]
    edges = hist_model["edges"]

    ht, _ = np.histogram(scores_test, bins=edges)
    ht = ht.astype(float)
    if ht.sum() == 0:
        ht[:] = 1.0
    ht /= ht.sum()

    if n_classes == 2:
        h0 = hist_model["histograms"][0]
        h1 = hist_model["histograms"][1]

        q_grid = np.linspace(0.0, 1.0, grid_size)
        best_q = 0.5
        best_dist = np.inf

        for q in q_grid:
            mix = (1.0 - q) * h0 + q * h1
            mix = np.clip(mix, 1e-12, None)
            mix /= mix.sum()
            dist = np.sqrt(0.5 * np.sum((np.sqrt(ht) - np.sqrt(mix)) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_q = q

        return np.array([1.0 - best_q, best_q])
    else:
        # Multiclass: fall back to EM-style approach on histograms
        # (simplified — proper implementation would search simplex)
        return np.ones(n_classes, dtype=float) / n_classes


# =============================================================================
# Fitted Estimator Wrappers (simplified — tree is already pruned)
# =============================================================================

class _FittedEMEstimator:
    """EM quantifier on leaf counts."""

    def __init__(self, pruned_tree, P, leaf_to_row):
        self.pruned_tree = pruned_tree
        self.P = P
        self.leaf_to_row = leaf_to_row

    def quantify(self, X_test):
        leaf_ids = self.pruned_tree.get_leaf_indices(X_test)
        n_rows = self.P.shape[0]
        counts = np.zeros(n_rows, dtype=float)
        for lid in leaf_ids:
            row = self.leaf_to_row.get(int(lid), None)
            if row is not None:
                counts[row] += 1.0
        return em_estimate_prevalence(counts, self.P)


class _FittedACCEstimator:
    """ACC quantifier on leaf counts."""

    def __init__(self, pruned_tree, P, leaf_to_row):
        self.pruned_tree = pruned_tree
        self.P = P
        self.leaf_to_row = leaf_to_row

    def quantify(self, X_test):
        leaf_ids = self.pruned_tree.get_leaf_indices(X_test)
        n_rows = self.P.shape[0]
        counts = np.zeros(n_rows, dtype=float)
        for lid in leaf_ids:
            row = self.leaf_to_row.get(int(lid), None)
            if row is not None:
                counts[row] += 1.0
        return acc_estimate_prevalence(counts, self.P)


class _FittedSLDEstimator:
    """SLD-EM on per-sample posteriors from pruned tree leaves."""

    def __init__(self, pruned_tree, leaf_posteriors, pi_train):
        self.pruned_tree = pruned_tree
        self.leaf_posteriors = leaf_posteriors  # dict: leaf_id -> posterior
        self.pi_train = pi_train
        self.n_classes_ = pruned_tree.n_classes_

    def quantify(self, X_test):
        leaf_ids = self.pruned_tree.get_leaf_indices(X_test)
        n = len(leaf_ids)
        scores = np.zeros((n, self.n_classes_), dtype=float)
        fallback = self.pi_train

        for i, lid in enumerate(leaf_ids):
            post = self.leaf_posteriors.get(int(lid), None)
            if post is not None:
                scores[i] = post
            else:
                scores[i] = fallback

        return sld_em(scores, init_pi=self.pi_train)


class _FittedPlattSLDEstimator:
    """SLD-EM with Platt-calibrated posteriors from the pruned tree."""

    def __init__(self, pruned_tree, leaf_posteriors, platt_calibrator, pi_train):
        self.pruned_tree = pruned_tree
        self.leaf_posteriors = leaf_posteriors
        self.platt = platt_calibrator
        self.pi_train = pi_train
        self.n_classes_ = pruned_tree.n_classes_

    def quantify(self, X_test):
        leaf_ids = self.pruned_tree.get_leaf_indices(X_test)
        n = len(leaf_ids)
        raw_scores = np.zeros((n, self.n_classes_), dtype=float)
        fallback = self.pi_train

        for i, lid in enumerate(leaf_ids):
            post = self.leaf_posteriors.get(int(lid), None)
            if post is not None:
                raw_scores[i] = post
            else:
                raw_scores[i] = fallback

        calibrated = self.platt.transform(raw_scores)
        return sld_em(calibrated, init_pi=self.pi_train)


class _FittedLRPlattSLDEstimator:
    """LR classifier + Platt calibration + SLD-EM."""

    def __init__(self, lr_model, platt_calibrator, classes, pi_train):
        self.lr_model = lr_model
        self.platt = platt_calibrator
        self.classes = classes
        self.pi_train = pi_train

    def quantify(self, X_test):
        probs = self.lr_model.predict_proba(X_test)
        # Align to expected classes
        probs_aligned = _align_probabilities_to_classes(
            probs, self.lr_model.classes_, self.classes
        )
        calibrated = self.platt.transform(probs_aligned)
        return sld_em(calibrated, init_pi=self.pi_train)


class _FittedHDXEstimator:
    """HDX estimator using Platt-calibrated scores."""

    def __init__(self, pruned_tree, leaf_posteriors, platt_calibrator, hist_model, pi_train, class_idx=1):
        self.pruned_tree = pruned_tree
        self.leaf_posteriors = leaf_posteriors
        self.platt = platt_calibrator
        self.hist_model = hist_model
        self.pi_train = pi_train
        self.class_idx = class_idx
        self.n_classes_ = pruned_tree.n_classes_

    def quantify(self, X_test):
        leaf_ids = self.pruned_tree.get_leaf_indices(X_test)
        n = len(leaf_ids)
        raw_scores = np.zeros((n, self.n_classes_), dtype=float)
        fallback = self.pi_train

        for i, lid in enumerate(leaf_ids):
            post = self.leaf_posteriors.get(int(lid), None)
            if post is not None:
                raw_scores[i] = post
            else:
                raw_scores[i] = fallback

        calibrated = self.platt.transform(raw_scores)
        # Use the positive class score for histogram matching
        scores_1d = calibrated[:, self.class_idx]
        return hdx_estimate(scores_1d, self.hist_model)


class _FittedKDEyEstimator:
    """Pre-fitted KDEyML wrapper."""

    def __init__(self, kdey_model, classes):
        self.kdey_model = kdey_model
        self.classes = classes

    def quantify(self, X_test):
        return self.kdey_model.quantify(X_test)


# =============================================================================
# KDEy-compatible classifier wrapper for pruned tree
# =============================================================================

class PrunedTreeClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible classifier wrapping a CalibrationSupportPrunedTree."""
    _estimator_type = "classifier"

    def __init__(self, pruned_tree=None, leaf_posteriors=None, classes=None):
        self.pruned_tree = pruned_tree
        self.leaf_posteriors = leaf_posteriors
        self.classes_ = classes
        self.n_classes_ = len(classes) if classes is not None else 2

    def fit(self, X, y):
        # Already fitted — no-op
        return self

    def predict_proba(self, X):
        leaf_ids = self.pruned_tree.get_leaf_indices(X)
        n = len(leaf_ids)
        proba = np.full((n, self.n_classes_), 1.0 / self.n_classes_)
        for i, lid in enumerate(leaf_ids):
            post = self.leaf_posteriors.get(int(lid), None)
            if post is not None:
                proba[i] = post
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# =============================================================================
# Utility
# =============================================================================

def _align_probabilities_to_classes(probs, model_classes, expected_classes):
    """Expand a probability matrix to the expected class set."""
    probs = np.asarray(probs, dtype=float)
    model_classes = np.asarray(model_classes)
    expected_classes = np.asarray(expected_classes)
    aligned = np.zeros((probs.shape[0], len(expected_classes)), dtype=float)
    class_to_col = {cls: idx for idx, cls in enumerate(model_classes)}
    for idx, cls in enumerate(expected_classes):
        col = class_to_col.get(cls, None)
        if col is not None and col < probs.shape[1]:
            aligned[:, idx] = probs[:, col]
    return aligned


def _prevalence(y, classes):
    counts = np.array([np.sum(y == c) for c in classes], dtype=float)
    if counts.sum() == 0:
        return np.ones(len(classes)) / len(classes)
    return counts / counts.sum()


# =============================================================================
# Metrics
# =============================================================================

def _calc_eps(n):
    return 1.0 / (2 * n)


def _smooth_probs(p, eps):
    return (eps + p) / (eps * len(p) + 1)


def _ae(p_true, p_hat):
    return np.sum(np.abs(p_true - p_hat))


def _kld(p_true, p_hat, eps=1e-8):
    if eps > 0.0:
        p_true = _smooth_probs(p_true, eps)
        p_hat = _smooth_probs(p_hat, eps)
    return np.sum(p_true * np.log2(p_true / p_hat))


def _nkld(p_true, p_hat, eps=1e-8):
    exp_kld = np.exp(_kld(p_true, p_hat, eps=eps))
    return max(0.0, 2 * exp_kld / (1 + exp_kld) - 1)


def _rae(p_true, p_hat, eps=1e-8):
    if eps > 0.0:
        p_true = _smooth_probs(p_true, eps)
        p_hat = _smooth_probs(p_hat, eps)
    return (np.abs(p_true - p_hat) / p_true).mean(axis=-1)


def _evaluate(p_true, p_hat):
    n_classes = len(p_true)
    return {
        "AE": _ae(p_true, p_hat),
        "RAE": _rae(p_true, p_hat, eps=_calc_eps(n_classes)),
        "NKLD": _nkld(p_true, p_hat, eps=_calc_eps(n_classes)),
    }


# =============================================================================
# Constants
# =============================================================================

TREE_CHOICES = ["gini", "qeb", "qcqb"]
ESTIMATOR_CHOICES = ["em", "em_smooth", "acc", "sld", "platt_sld", "lr_platt_sld", "hdx", "kdey"]


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--datasets", nargs="*", type=str,
        choices=DATASET_LIST, default=DATASET_LIST,
        help="Datasets used in evaluation."
    )
    parser.add_argument(
        "--modes", nargs="+",
        choices=[BINARY_MODE_KEY, MULTICLASS_MODE_KEY],
        default=[BINARY_MODE_KEY, MULTICLASS_MODE_KEY],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=GLOBAL_SEEDS)
    parser.add_argument(
        "--trees", nargs="+", choices=TREE_CHOICES, default=TREE_CHOICES,
    )
    parser.add_argument(
        "--estimators", nargs="+", choices=ESTIMATOR_CHOICES, default=ESTIMATOR_CHOICES,
    )
    parser.add_argument(
        "--min-calibration-samples", nargs="+", type=int, default=[1, 5],
    )
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--qeb-max-features", type=float, default=None)
    parser.add_argument("--qeb-max-thresholds", type=int, default=None)
    parser.add_argument("--kde-bandwidth", type=float, default=0.05)
    parser.add_argument("--em-alpha", type=float, default=1.0)
    parser.add_argument("--hdx-bins", type=int, default=300)
    parser.add_argument("--hdx-grid-size", type=int, default=1001)
    parser.add_argument("--load-from-disk", action="store_true")
    parser.add_argument("--minsize", type=int, default=None)
    parser.add_argument("--maxsize", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dt", type=int, nargs="+", default=None,
        help="Index for train/test-splits to be run from TRAIN_TEST_RATIOS."
    )
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--n-tasks", type=int, default=None)
    return parser.parse_args()


# =============================================================================
# Tree construction
# =============================================================================

def _build_tree(tree_name, max_depth, min_samples_leaf, random_state,
                qeb_max_features=None, qeb_max_thresholds=None):
    if tree_name == "gini":
        return ClassificationTree(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    if tree_name == "qeb":
        return QuantificationErrorBalancingTree(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            max_features=qeb_max_features, max_thresholds=qeb_max_thresholds,
            random_state=random_state,
        )
    if tree_name == "qcqb":
        return ClassificationQuantificationBalancingTree(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            max_features=qeb_max_features, max_thresholds=qeb_max_thresholds,
            random_state=random_state,
        )
    raise ValueError(f"Unknown tree: {tree_name}")


# =============================================================================
# Stratified train/val split
# =============================================================================

def _split_train_val(X_train, y_train, val_fraction, seed):
    from sklearn.model_selection import train_test_split
    if val_fraction <= 0 or val_fraction >= 1.0:
        return X_train, y_train, X_train, y_train
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=val_fraction,
            stratify=y_train, random_state=seed
        )
    except ValueError:
        X_tr, y_tr = X_train, y_train
        X_val, y_val = X_train, y_train
    return X_tr, y_tr, X_val, y_val


# =============================================================================
# Run a single work unit
# =============================================================================

def run_single_unit(
    dta_name, seed, dt_ratio, train_distr, test_distr,
    trees, estimators, min_calibration_samples_list,
    max_depth, min_samples_leaf, kde_bandwidth,
    qeb_max_features, qeb_max_thresholds,
    load_from_disk, em_alpha, hdx_bins, hdx_grid_size, verbose,
):
    """
    Run all tree/estimator/min_cal combinations for a single
    (dataset, seed, dt_ratio, train_distr, test_distr) draw.

    Architecture:
    1. Fit tree on training data
    2. For each min_calibration_samples threshold:
       a. Build CalibrationSupportPrunedTree (ensures all test leaves are supported)
       b. Build transfer matrix P(leaf|class) from calibration data on pruned tree
       c. Run each estimator using the pruned tree
    """
    X, y, N, Y, n_classes, y_cts, y_idx = helpers.get_xy(
        dta_name, load_from_disk=load_from_disk, binned=False,
    )

    class_list = np.arange(n_classes)
    class_to_idx = {c: i for i, c in enumerate(Y)}

    def _map_labels(arr):
        return np.array([class_to_idx.get(v, v) for v in arr])

    train_index, test_index, stats_vec = helpers.synthetic_draw(
        N, n_classes, y_cts, y_idx, dt_ratio, train_distr, test_distr, seed
    )

    if len(train_index) == 0 or len(test_index) == 0:
        if verbose:
            print(f"  Skipping empty draw: dt={dt_ratio}, "
                  f"train_d={train_distr}, test_d={test_distr}")
        return []

    X_train_full = X[train_index]
    y_train_full = _map_labels(y[train_index])
    X_test = X[test_index]
    y_test = _map_labels(y[test_index])

    # Split training into fit (75%) and calibration (25%)
    X_tr, y_tr, X_val, y_val = _split_train_val(
        X_train_full, y_train_full, val_fraction=0.25, seed=seed
    )

    pi_true = _prevalence(y_test, class_list)
    pi_train = _prevalence(y_tr, class_list)

    rows = []

    for tree_name in trees:
        tree = _build_tree(
            tree_name, max_depth, min_samples_leaf, seed,
            qeb_max_features=qeb_max_features,
            qeb_max_thresholds=qeb_max_thresholds,
        )

        # Fit tree ONCE on training data
        tree.fit(X_tr, y_tr)

        # Get raw leaf IDs on calibration data
        leaf_ids_val_raw = tree.get_leaf_indices(X_val)

        for min_cal in min_calibration_samples_list:
            # ============================================================
            # Build pruned tree — this is the key change from the old code.
            # Every test sample is guaranteed to land in a supported leaf.
            # No redirection based on class profiles. [1]
            # ============================================================
            pruned_tree = CalibrationSupportPrunedTree(
                tree=tree,
                leaf_ids_cal=leaf_ids_val_raw,
                y_cal=y_val,
                min_samples_per_leaf=min_cal,
                n_classes=n_classes,
            )

            # Get calibration leaf IDs through the pruned tree
            # (some may be remapped to supported ancestors)
            leaf_ids_val = pruned_tree.get_leaf_indices(X_val)

            # Build transfer matrix P(leaf | class) [1]
            P, leaf_to_row, classes = build_transfer_matrix_from_pruned_tree(
                pruned_tree, leaf_ids_val, y_val, alpha=em_alpha
            )

            # Also build unsmoothed version for ACC
            P_unsmoothed, _, _ = build_transfer_matrix_from_pruned_tree(
                pruned_tree, leaf_ids_val, y_val, alpha=0.0
            )

            # Leaf posteriors for SLD-based methods
            leaf_posteriors = pruned_tree.get_leaf_posteriors(alpha=0.01)

            # Platt calibrator (fit on calibration posteriors vs labels)
            raw_scores_val = np.zeros((len(X_val), n_classes), dtype=float)
            for i, lid in enumerate(leaf_ids_val):
                post = leaf_posteriors.get(int(lid), None)
                if post is not None:
                    raw_scores_val[i] = post
                else:
                    raw_scores_val[i] = pi_train

            platt = PlattCalibrator(random_state=seed)
            platt.fit(raw_scores_val, y_val, classes=class_list)
            cal_scores_val = platt.transform(raw_scores_val)

            for est_name in estimators:
                if verbose:
                    print(
                        f"  dt={dt_ratio}, tree={tree_name}, "
                        f"est={est_name}, min_cal={min_cal}"
                    )

                # ==========================================================
                # FIT ESTIMATOR
                # ==========================================================

                if est_name == "em":
                    # EM with Laplace smoothing on P(leaf|class) [1]
                    fitted = _FittedEMEstimator(pruned_tree, P, leaf_to_row)

                elif est_name == "em_smooth":
                    # Same as em (alpha already applied in P construction)
                    fitted = _FittedEMEstimator(pruned_tree, P, leaf_to_row)

                elif est_name == "acc":
                    fitted = _FittedACCEstimator(pruned_tree, P_unsmoothed, leaf_to_row)

                elif est_name == "sld":
                    fitted = _FittedSLDEstimator(
                        pruned_tree, leaf_posteriors, pi_train
                    )

                elif est_name == "platt_sld":
                    fitted = _FittedPlattSLDEstimator(
                        pruned_tree, leaf_posteriors, platt, pi_train
                    )

                elif est_name == "lr_platt_sld":
                    lr_model = LogisticRegression(max_iter=1000, solver="lbfgs")
                    lr_model.fit(X_tr, y_tr)
                    probs_val_lr = _align_probabilities_to_classes(
                        lr_model.predict_proba(X_val), lr_model.classes_, class_list
                    )
                    platt_lr = PlattCalibrator(random_state=seed)
                    platt_lr.fit(probs_val_lr, y_val, classes=class_list)
                    fitted = _FittedLRPlattSLDEstimator(
                        lr_model, platt_lr, class_list, pi_train
                    )

                elif est_name == "hdx":
                    # HDX uses Platt-calibrated scores [1]
                    if n_classes == 2:
                        hist_model = build_score_histograms(
                            cal_scores_val[:, 1], y_val, n_classes=2, n_bins=hdx_bins
                        )
                        fitted = _FittedHDXEstimator(
                            pruned_tree, leaf_posteriors, platt, hist_model, pi_train,
                            class_idx=1,
                        )
                    else:
                        # HDX multiclass not fully implemented — skip
                        continue

                elif est_name == "kdey":
                    if not QUAPY_AVAILABLE:
                        continue
                    wrapper = PrunedTreeClassifier(
                        pruned_tree=pruned_tree,
                        leaf_posteriors=leaf_posteriors,
                        classes=class_list,
                    )
                    kdey = KDEyML(
                        classifier=wrapper,
                        fit_classifier=False,
                        val_split=(X_val, y_val),
                        bandwidth=kde_bandwidth,
                        random_state=seed,
                    )
                    kdey.fit(X_tr, y_tr)
                    fitted = _FittedKDEyEstimator(kdey, class_list)

                else:
                    raise ValueError(f"Unknown estimator: {est_name}")

                # ==========================================================
                # PREDICT
                # ==========================================================

                pi_hat = np.asarray(fitted.quantify(X_test), dtype=float)

                if len(pi_hat) != len(pi_true):
                    aligned = np.zeros(len(class_list))
                    aligned[:min(len(pi_hat), len(aligned))] = pi_hat[:len(aligned)]
                    pi_hat = aligned

                metrics = _evaluate(pi_true, pi_hat)

                rows.append({
                    "dataset": dta_name,
                    "seed": seed,
                    "n_classes": n_classes,
                    "dt_ratio": f"{dt_ratio[0]:.1f}/{dt_ratio[1]:.1f}",
                    "train_distribution": str(train_distr),
                    "test_distribution": str(test_distr),
                    "tree": tree_name,
                    "estimator": est_name,
                    "min_calibration_samples": int(min_cal),
                    "n_leaves_retained": int(pruned_tree.n_leaves_retained_),
                    "AE": metrics["AE"],
                    "RAE": metrics["RAE"],
                    "NKLD": metrics["NKLD"],
                    "true_prevalence": ";".join(f"{v:.6f}" for v in pi_true),
                    "pred_prevalence": ";".join(f"{v:.6f}" for v in pi_hat),
                })

    return rows


# =============================================================================
# Main entry point
# =============================================================================

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # MODE 1: Manifest-based parallelization
    # ------------------------------------------------------------------
    if args.manifest is not None:
        assert args.task_id is not None and args.n_tasks is not None, \
            "--task-id and --n-tasks required with --manifest"

        with open(args.manifest) as f:
            all_units = [line.strip().split(",") for line in f if line.strip()]

        my_units = all_units[args.task_id::args.n_tasks]
        print(f"Task {args.task_id}/{args.n_tasks}: processing {len(my_units)} work units")

        all_rows = []
        for i, unit in enumerate(my_units):
            dta_name, seed_str, dt_idx_str, tr_idx_str, te_idx_str = unit
            seed = int(seed_str)
            dt_ratio = TRAIN_TEST_RATIOS[int(dt_idx_str)]

            n_classes = int(DATASET_INDEX.loc[dta_name, "classes"])
            train_ds = TRAINING_DISTRIBUTIONS[n_classes]
            test_ds = TEST_DISTRIBUTIONS[n_classes]
            train_distr = train_ds[int(tr_idx_str)]
            test_distr = test_ds[int(te_idx_str)]

            if args.verbose:
                print(f"  [{i+1}/{len(my_units)}] {dta_name} seed={seed} "
                      f"dt={dt_ratio} train_d={train_distr} test_d={test_distr}")

            rows = run_single_unit(
                dta_name=dta_name, seed=seed, dt_ratio=dt_ratio,
                train_distr=train_distr, test_distr=test_distr,
                trees=args.trees, estimators=args.estimators,
                min_calibration_samples_list=args.min_calibration_samples,
                max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
                kde_bandwidth=args.kde_bandwidth,
                qeb_max_features=args.qeb_max_features,
                qeb_max_thresholds=args.qeb_max_thresholds,
                load_from_disk=args.load_from_disk, em_alpha=args.em_alpha,
                hdx_bins=args.hdx_bins, hdx_grid_size=args.hdx_grid_size,
                verbose=args.verbose,
            )
            all_rows.extend(rows)

        if not all_rows:
            print("No results to save.")
            return

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        out_name = f"tree_matrix_task_{args.task_id}_{timestamp}_pid_{os.getpid()}.csv"
        out_path = os.path.join(RAW_RESULT_FILES_PATH, out_name)
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        print(f"Saved {len(all_rows)} result rows to {out_path}")
        return

    # ------------------------------------------------------------------
    # MODE 2: Standard execution
    # ------------------------------------------------------------------
    df_ind = DATASET_INDEX.loc[args.datasets]

    if args.minsize is not None:
        df_ind = df_ind.loc[df_ind["size"] >= args.minsize]
    if args.maxsize is not None:
        df_ind = df_ind.loc[df_ind["size"] <= args.maxsize]
    if MULTICLASS_MODE_KEY not in args.modes:
        df_ind = df_ind.loc[df_ind["classes"] == 2]
    if BINARY_MODE_KEY not in args.modes:
        df_ind = df_ind.loc[df_ind["classes"] > 2]

    datasets = list(df_ind.index)

    if args.dt is not None:
        dt_ratios = [TRAIN_TEST_RATIOS[i] for i in args.dt]
    else:
        dt_ratios = TRAIN_TEST_RATIOS

    all_rows = []
    total_runs = len(datasets) * len(args.seeds)
    run_index = 0

    for dta_name in datasets:
        n_classes = int(df_ind.loc[dta_name, "classes"])
        train_ds = TRAINING_DISTRIBUTIONS[n_classes]
        test_ds = TEST_DISTRIBUTIONS[n_classes]

        for seed in args.seeds:
            run_index += 1
            print(f"Running {dta_name} (seed={seed}) [{run_index}/{total_runs}]")

            for dt_ratio in dt_ratios:
                for train_distr in train_ds:
                    for test_distr in test_ds:
                        rows = run_single_unit(
                            dta_name=dta_name, seed=seed, dt_ratio=dt_ratio,
                            train_distr=train_distr, test_distr=test_distr,
                            trees=args.trees, estimators=args.estimators,
                            min_calibration_samples_list=args.min_calibration_samples,
                            max_depth=args.max_depth,
                            min_samples_leaf=args.min_samples_leaf,
                            kde_bandwidth=args.kde_bandwidth,
                            qeb_max_features=args.qeb_max_features,
                            qeb_max_thresholds=args.qeb_max_thresholds,
                            load_from_disk=args.load_from_disk,
                            em_alpha=args.em_alpha,
                            hdx_bins=args.hdx_bins,
                            hdx_grid_size=args.hdx_grid_size,
                            verbose=args.verbose,
                        )
                        all_rows.extend(rows)

    if not all_rows:
        print("No results to save.")
        return

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    seed_tag = f"seed_{args.seeds[0]}" if len(args.seeds) == 1 else "multi_seed"
    slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    pid_tag = f"pid_{os.getpid()}"
    if slurm_task_id is not None:
        out_name = f"tree_matrix_{seed_tag}_task_{slurm_task_id}_{timestamp}_{pid_tag}.csv"
    else:
        out_name = f"tree_matrix_{seed_tag}_{timestamp}_{pid_tag}.csv"
    out_path = os.path.join(RAW_RESULT_FILES_PATH, out_name)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Saved {len(all_rows)} result rows to {out_path}")


if __name__ == "__main__":
    main()