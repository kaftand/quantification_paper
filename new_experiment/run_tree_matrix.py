import argparse
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

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
    HoldoutTransferMatrixEstimator,
)

import quapy as qp
from quapy.method.aggregative import ACC, EMQ, KDEyML
from quapy.data import LabelledCollection


# =============================================================================
# Pruned Tree Classifier (sklearn-compatible wrapper)
# =============================================================================

class PrunedTreeClassifier(ClassifierMixin, BaseEstimator):
    """
    Sklearn-compatible classifier wrapping a fitted tree with calibration-support pruning.

    Guarantees every sample lands in a leaf with calibration support.
    Posteriors are computed from calibration label counts per leaf (smoothed).
    """
    _estimator_type = "classifier"

    def __init__(self, tree=None, min_calibration_samples=1, alpha=1.0, n_classes=None):
        self.tree = tree
        self.min_calibration_samples = min_calibration_samples
        self.alpha = alpha
        self.n_classes = n_classes

    def fit(self, X, y):
        """
        Fit the tree on X, y. Calibration (pruning + posteriors) happens
        separately via calibrate().
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if self.n_classes is not None:
            self.classes_ = np.arange(self.n_classes)
        else:
            self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.tree.fit(X, y)
        self._is_calibrated = False
        return self

    def calibrate(self, X_val, y_val):
        """
        Calibrate: determine supported leaves and compute posteriors.
        Must be called after fit() and before predict/predict_proba on test data.
        """
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)

        leaf_ids_val = self.tree.get_leaf_indices(X_val)

        # Determine which leaves have enough calibration support
        leaves, counts = np.unique(leaf_ids_val, return_counts=True)
        self.supported_leaves_ = set(
            int(leaf) for leaf, count in zip(leaves, counts)
            if count >= self.min_calibration_samples
        )

        # Build per-leaf label counts (only for supported leaves)
        self._leaf_label_counts = {}
        for leaf_id, label in zip(leaf_ids_val, y_val):
            lid = int(leaf_id)
            if lid in self.supported_leaves_:
                if lid not in self._leaf_label_counts:
                    self._leaf_label_counts[lid] = np.zeros(self.n_classes_, dtype=float)
                if 0 <= int(label) < self.n_classes_:
                    self._leaf_label_counts[lid][int(label)] += 1.0

        # Compute smoothed posteriors per leaf
        self._leaf_posteriors = {}
        for lid, counts_vec in self._leaf_label_counts.items():
            self._leaf_posteriors[lid] = (
                (counts_vec + self.alpha) /
                (counts_vec.sum() + self.alpha * self.n_classes_)
            )

        # Fallback: for unsupported leaves, find nearest supported leaf
        # Simple approach: map to the supported leaf with most samples
        if self.supported_leaves_:
            self._fallback_leaf = max(
                self._leaf_label_counts.keys(),
                key=lambda lid: self._leaf_label_counts[lid].sum()
            )
        else:
            self._fallback_leaf = None

        self._is_calibrated = True
        return self

    def _resolve_leaf(self, leaf_id):
        """Map a raw leaf ID to a supported leaf ID."""
        lid = int(leaf_id)
        if lid in self.supported_leaves_:
            return lid
        return self._fallback_leaf

    def predict_proba(self, X):
        X = np.asarray(X)
        leaf_ids = self.tree.get_leaf_indices(X)
        n = len(leaf_ids)
        proba = np.full((n, self.n_classes_), 1.0 / self.n_classes_)

        if not self._is_calibrated:
            return proba

        for i, lid in enumerate(leaf_ids):
            resolved = self._resolve_leaf(lid)
            if resolved is not None and resolved in self._leaf_posteriors:
                proba[i] = self._leaf_posteriors[resolved]

        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# =============================================================================
# Transfer Matrix + EM (our custom leaf-based EM from v6) [1]
# =============================================================================

def build_transfer_matrix(tree, X_val, y_val, n_classes, min_calibration_samples=1, alpha=1.0):
    """
    Build P(leaf | class) from calibration data, only using supported leaves.
    Returns P, leaf_to_row mapping.
    """
    leaf_ids_val = tree.get_leaf_indices(X_val)
    leaves, counts = np.unique(leaf_ids_val, return_counts=True)

    supported_leaves = sorted(
        int(leaf) for leaf, count in zip(leaves, counts)
        if count >= min_calibration_samples
    )

    if len(supported_leaves) == 0:
        # Degenerate case
        return np.ones((1, n_classes)) / n_classes, {0: 0}, {0}

    leaf_to_row = {leaf: i for i, leaf in enumerate(supported_leaves)}
    n_leaves = len(supported_leaves)
    supported_set = set(supported_leaves)

    # Count per (leaf, class), only for supported leaves
    counts_matrix = np.zeros((n_leaves, n_classes), dtype=float)
    for leaf_id, label in zip(leaf_ids_val, y_val):
        lid = int(leaf_id)
        if lid in supported_set and 0 <= int(label) < n_classes:
            counts_matrix[leaf_to_row[lid], int(label)] += 1.0

    # Normalize per class: P(leaf | class) [1][3]
    P = np.zeros((n_leaves, n_classes), dtype=float)
    for k in range(n_classes):
        col_sum = counts_matrix[:, k].sum()
        if col_sum > 0:
            P[:, k] = (counts_matrix[:, k] + alpha) / (col_sum + alpha * n_leaves)
        else:
            P[:, k] = np.ones(n_leaves) / n_leaves

    return P, leaf_to_row, supported_set


def em_estimate_prevalence(counts_vec, P, init_pi=None, tol=1e-8, max_iter=500):
    """
    EM on leaf counts with P(leaf|class) matrix. [1]
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

    for _ in range(1, max_iter + 1):
        mix = P @ pi
        mix = np.clip(mix, 1e-12, None)
        R = (P * pi[None, :]) / mix[:, None]
        pi_new = (counts_vec[:, None] * R).sum(axis=0) / total
        pi_new = np.clip(pi_new, 1e-12, None)
        pi_new /= pi_new.sum()

        if np.sum(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new

    return pi


class _FittedLeafEM:
    """Leaf-count EM estimator (the QTree-EM approach from v6) [1]."""

    def __init__(self, tree, P, leaf_to_row, supported_set, fallback_leaf=None):
        self.tree = tree
        self.P = P
        self.leaf_to_row = leaf_to_row
        self.supported_set = supported_set
        self.fallback_leaf = fallback_leaf

    def quantify(self, X_test):
        leaf_ids = self.tree.get_leaf_indices(X_test)
        n_rows = self.P.shape[0]
        counts = np.zeros(n_rows, dtype=float)

        for lid in leaf_ids:
            lid_int = int(lid)
            if lid_int in self.leaf_to_row:
                counts[self.leaf_to_row[lid_int]] += 1.0
            elif self.fallback_leaf is not None and self.fallback_leaf in self.leaf_to_row:
                counts[self.leaf_to_row[self.fallback_leaf]] += 1.0
            # else: skip (like v6 does) [1]

        return em_estimate_prevalence(counts, self.P)


# =============================================================================
# HDX Estimator [1]
# =============================================================================

def build_score_histograms(scores, y, n_classes, n_bins=300):
    """Build per-class score histograms from calibration scores."""
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
    """Binary HDX prevalence estimate via Hellinger distance minimization. [1]"""
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
        best_q, best_dist = 0.5, np.inf

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
        return np.ones(n_classes) / n_classes


class _FittedHDX:
    """Pre-fitted HDX estimator."""

    def __init__(self, clf, hist_model, grid_size=1001, class_idx=1):
        self.clf = clf
        self.hist_model = hist_model
        self.grid_size = grid_size
        self.class_idx = class_idx

    def quantify(self, X_test):
        proba = self.clf.predict_proba(X_test)
        scores = proba[:, self.class_idx]
        return hdx_estimate(scores, self.hist_model, self.grid_size)


# =============================================================================
# QuaPy Wrappers
# =============================================================================

class _FittedQuaPyEstimator:
    """Wrapper for a fitted QuaPy quantifier."""

    def __init__(self, quapy_model, classes):
        self.quapy_model = quapy_model
        self.classes = classes

    def quantify(self, X_test):
        # QuaPy returns prevalence in the order of its internal classes
        return self.quapy_model.quantify(X_test)


def _fit_quapy_acc(clf, X_val, y_val, classes):
    """
    Fit QuaPy ACC using the pruned tree classifier.
    val_split=(X_val, y_val) tells ACC to use this specific data
    for computing the misclassification matrix.
    """
    acc = ACC(
        classifier=clf,
        fit_classifier=False,
        val_split=(X_val, y_val),
        solver='minimize',
        norm='clip',
    )
    # QuaPy needs a LabelledCollection for fit, but with fit_classifier=False
    # and val_split as tuple, it just computes the confusion matrix from val_split.
    # We still need to call fit() — it won't retrain the classifier.
    lc_dummy = LabelledCollection(X_val, y_val, classes=classes)
    acc.fit(lc_dummy)
    return acc


def _fit_quapy_emq(clf, X_val, y_val, classes, exact_train_prev=True):
    """
    Fit QuaPy EMQ (SLD) using the pruned tree classifier.
    val_split=(X_val, y_val) provides the data for estimating training prevalence
    and calibrating posteriors.
    """
    emq = EMQ(
        classifier=clf,
        fit_classifier=False,
        val_split=(X_val, y_val),
        exact_train_prev=exact_train_prev,
    )
    lc_dummy = LabelledCollection(X_val, y_val, classes=classes)
    emq.fit(lc_dummy)
    return emq


def _fit_quapy_kdey(clf, X_val, y_val, classes, bandwidth=0.05, random_state=None):
    """Fit QuaPy KDEyML using the pruned tree classifier."""
    kdey = KDEyML(
        classifier=clf,
        fit_classifier=False,
        val_split=(X_val, y_val),
        bandwidth=bandwidth,
        random_state=random_state,
    )
    lc_dummy = LabelledCollection(X_val, y_val, classes=classes)
    kdey.fit(lc_dummy)
    return kdey


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


def _prevalence(y, classes):
    counts = np.array([np.sum(y == c) for c in classes], dtype=float)
    if counts.sum() == 0:
        return np.ones(len(classes)) / len(classes)
    return counts / counts.sum()


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
ESTIMATOR_CHOICES = ["leaf_em", "acc", "sld", "sld_bcts", "hdx", "kdey"]


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--datasets", nargs="*", type=str,
        choices=DATASET_LIST, default=DATASET_LIST,
    )
    parser.add_argument(
        "--modes", nargs="+",
        choices=[BINARY_MODE_KEY, MULTICLASS_MODE_KEY],
        default=[BINARY_MODE_KEY, MULTICLASS_MODE_KEY],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=GLOBAL_SEEDS)
    parser.add_argument("--trees", nargs="+", choices=TREE_CHOICES, default=TREE_CHOICES)
    parser.add_argument("--estimators", nargs="+", choices=ESTIMATOR_CHOICES, default=ESTIMATOR_CHOICES)
    parser.add_argument("--min-calibration-samples", nargs="+", type=int, default=[1, 5])
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
    parser.add_argument("--dt", type=int, nargs="+", default=None)
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
    Run all tree/estimator/min_cal combinations for a single experimental draw.

    Architecture:
    1. Fit tree on training data
    2. Calibrate (determine supported leaves + posteriors) on validation data
    3. For ACC/SLD: use QuaPy with the calibrated tree as classifier
    4. For leaf_em: use our EM on P(leaf|class) directly [1]
    5. For hdx: use histogram matching on calibrated scores [1]
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
            print(f"  Skipping empty draw")
        return []

    X_train_full = X[train_index]
    y_train_full = _map_labels(y[train_index])
    X_test = X[test_index]
    y_test = _map_labels(y[test_index])

    # Split: 75% train, 25% calibration
    X_tr, y_tr, X_val, y_val = _split_train_val(
        X_train_full, y_train_full, val_fraction=0.5, seed=seed
    )

    pi_true = _prevalence(y_test, class_list)

    rows = []

    for tree_name in trees:
        raw_tree = _build_tree(
            tree_name, max_depth, min_samples_leaf, seed,
            qeb_max_features=qeb_max_features,
            qeb_max_thresholds=qeb_max_thresholds,
        )

        # Fit tree ONCE
        raw_tree.fit(X_tr, y_tr)

        for min_cal in min_calibration_samples_list:
            # Build calibrated classifier
            clf = PrunedTreeClassifier(
                tree=raw_tree,
                min_calibration_samples=min_cal,
                alpha=em_alpha,
                n_classes=n_classes,
            )
            # We already fitted the tree inside PrunedTreeClassifier.fit would
            # refit. Instead, manually set up the classifier state:
            clf.classes_ = class_list
            clf.n_classes_ = n_classes
            clf._is_calibrated = False
            clf.tree = raw_tree  # already fitted

            # Calibrate on validation data
            clf.calibrate(X_val, y_val)

            # Build transfer matrix for leaf-EM [1]
            P, leaf_to_row, supported_set = build_transfer_matrix(
                raw_tree, X_val, y_val, n_classes,
                min_calibration_samples=min_cal, alpha=em_alpha,
            )
            fallback_leaf = (
                max(clf._leaf_label_counts.keys(),
                    key=lambda lid: clf._leaf_label_counts[lid].sum())
                if clf._leaf_label_counts else None
            )

            for est_name in estimators:
                if verbose:
                    print(
                        f"  dt={dt_ratio}, tree={tree_name}, "
                        f"est={est_name}, min_cal={min_cal}"
                    )

                try:
                    if est_name == "leaf_em":
                        # Our custom leaf-count EM [1]
                        fitted = _FittedLeafEM(
                            raw_tree, P, leaf_to_row, supported_set, fallback_leaf
                        )
                        pi_hat = fitted.quantify(X_test)

                    elif est_name == "acc":
                        # QuaPy ACC — uses confusion matrix from val data
                        acc_model = _fit_quapy_acc(clf, X_val, y_val, class_list)
                        pi_hat = acc_model.quantify(X_test)

                    elif est_name == "sld":
                        # QuaPy EMQ (SLD) — uses classifier posteriors correctly
                        emq_model = _fit_quapy_emq(
                            clf, X_val, y_val, class_list, exact_train_prev=True
                        )
                        pi_hat = emq_model.quantify(X_test)

                    elif est_name == "sld_bcts":
                        # QuaPy EMQ with BCTS calibration
                        emq_model = EMQ(
                            classifier=clf,
                            fit_classifier=False,
                            val_split=(X_val, y_val),
                            exact_train_prev=True,
                            calib='bcts',
                        )
                        lc_dummy = LabelledCollection(X_val, y_val, classes=class_list)
                        emq_model.fit(lc_dummy)
                        pi_hat = emq_model.quantify(X_test)

                    elif est_name == "hdx":
                        # HDX on calibrated scores [1]
                        if n_classes != 2:
                            continue
                        scores_val = clf.predict_proba(X_val)[:, 1]
                        hist_model = build_score_histograms(
                            scores_val, y_val, n_classes=2, n_bins=hdx_bins
                        )
                        fitted_hdx = _FittedHDX(clf, hist_model, hdx_grid_size, class_idx=1)
                        pi_hat = fitted_hdx.quantify(X_test)

                    elif est_name == "kdey":
                        # QuaPy KDEyML
                        kdey_model = _fit_quapy_kdey(
                            clf, X_val, y_val, class_list,
                            bandwidth=kde_bandwidth, random_state=seed,
                        )
                        pi_hat = kdey_model.quantify(X_test)

                    else:
                        raise ValueError(f"Unknown estimator: {est_name}")

                except Exception as e:
                    if verbose:
                        print(f"    ERROR: {e}")
                    continue

                pi_hat = np.asarray(pi_hat, dtype=float)
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
                    "n_leaves_retained": len(clf.supported_leaves_),
                    "AE": metrics["AE"],
                    "RAE": metrics["RAE"],
                    "NKLD": metrics["NKLD"],
                    "true_prevalence": ";".join(f"{v:.6f}" for v in pi_true),
                    "pred_prevalence": ";".join(f"{v:.6f}" for v in pi_hat),
                })

    return rows


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # MODE 1: Manifest-based parallelization
    # ------------------------------------------------------------------
    if args.manifest is not None:
        assert args.task_id is not None and args.n_tasks is not None

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
                print(f"  [{i+1}/{len(my_units)}] {dta_name} seed={seed}")

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