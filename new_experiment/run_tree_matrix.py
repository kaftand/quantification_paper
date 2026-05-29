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
    EMQuantificationSolver,
    ACCQuantificationSolver,
)

try:
    from quapy.method.aggregative import KDEyML
    QUAPY_AVAILABLE = True
except ImportError:
    QUAPY_AVAILABLE = False


# =============================================================================
# TreePosteriorClassifier
# =============================================================================

class TreePosteriorClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible classifier using tree leaf posteriors as probability estimates."""
    _estimator_type = "classifier"

    def __init__(self, tree=None, alpha=0.01, min_calibration_samples=0, expected_classes=None):
        self.tree = tree
        self.alpha = alpha
        self.min_calibration_samples = min_calibration_samples
        self.expected_classes = expected_classes

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.asarray(self.expected_classes) if self.expected_classes is not None else np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.tree.fit(X, y)
        leaf_ids = self.tree.get_leaf_indices(X)

        leaf_ids_filtered = np.asarray(leaf_ids)
        y_filtered = np.asarray(y)
        self.leaf_redirect_ = {}

        if self.min_calibration_samples > 0:
            counts_per_leaf = np.bincount(leaf_ids_filtered.astype(int))
            observed_leaves = np.unique(leaf_ids_filtered)
            keep_leaves = set(
                leaf for leaf in observed_leaves
                if counts_per_leaf[int(leaf)] >= self.min_calibration_samples
            )
            pruned_leaves = set(
                leaf for leaf in observed_leaves if leaf not in keep_leaves
            )

            if len(pruned_leaves) > 0 and len(keep_leaves) > 0:
                n_classes = len(self.classes_)
                class_to_idx = {c: i for i, c in enumerate(self.classes_)}
                supported_leaf_list = sorted(keep_leaves)
                supported_profiles = np.zeros((len(supported_leaf_list), n_classes), dtype=float)
                for j, leaf in enumerate(supported_leaf_list):
                    mask = leaf_ids_filtered == leaf
                    for label in y_filtered[mask]:
                        idx = class_to_idx.get(label, None)
                        if idx is not None:
                            supported_profiles[j, idx] += 1
                    total = supported_profiles[j].sum()
                    if total > 0:
                        supported_profiles[j] /= total

                for leaf in pruned_leaves:
                    mask = leaf_ids_filtered == leaf
                    profile = np.zeros(n_classes, dtype=float)
                    for label in y_filtered[mask]:
                        idx = class_to_idx.get(label, None)
                        if idx is not None:
                            profile[idx] += 1
                    total = profile.sum()
                    if total > 0:
                        profile /= total
                    dists = np.linalg.norm(supported_profiles - profile[np.newaxis, :], axis=1)
                    self.leaf_redirect_[leaf] = supported_leaf_list[int(np.argmin(dists))]

            keep_mask = np.array([leaf in keep_leaves for leaf in leaf_ids_filtered], dtype=bool)
            leaf_ids_filtered = leaf_ids_filtered[keep_mask]
            y_filtered = y_filtered[keep_mask]

        estimator = HoldoutTransferMatrixEstimator(alpha=self.alpha)
        estimator.fit(leaf_ids_filtered, y_filtered, classes=self.expected_classes)
        self.P_ = estimator.P_
        self.leaf_to_row_ = estimator.leaf_to_row

        row_sums = self.P_.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        self.leaf_posteriors_ = self.P_ / row_sums
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        leaf_ids = self.tree.get_leaf_indices(X)
        n = len(leaf_ids)
        proba = np.full((n, self.n_classes_), 1.0 / self.n_classes_)
        for i, lid in enumerate(leaf_ids):
            row = self.leaf_to_row_.get(lid, None)
            if row is None:
                redirected = self.leaf_redirect_.get(lid, None)
                if redirected is not None:
                    row = self.leaf_to_row_.get(redirected, None)
            if row is not None:
                proba[i] = self.leaf_posteriors_[row]
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# =============================================================================
# Constants
# =============================================================================

TREE_CHOICES = ["gini", "qeb", "qcqb"]
ESTIMATOR_CHOICES = ["em", "em_smooth", "acc", "sld", "platt_sld", "lr_platt_sld", "kdey"]


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
        "--min-calibration-samples", nargs="+", type=int, default=[0, 5],
    )
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--qeb-max-features", type=float, default=None)
    parser.add_argument("--qeb-max-thresholds", type=int, default=None)
    parser.add_argument("--kde-bandwidth", type=float, default=0.05)
    parser.add_argument("--em-alpha", type=float, default=1.0)
    parser.add_argument("--load-from-disk", action="store_true")
    parser.add_argument("--minsize", type=int, default=None)
    parser.add_argument("--maxsize", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dt", type=int, nargs="+", default=None,
        help="Index for train/test-splits to be run from TRAIN_TEST_RATIOS."
    )
    # Manifest-based parallelization
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to manifest file for parallel execution.")
    parser.add_argument("--task-id", type=int, default=None,
                        help="SLURM array task ID for round-robin assignment.")
    parser.add_argument("--n-tasks", type=int, default=None,
                        help="Total number of parallel tasks.")
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
# Transfer matrix helpers
# =============================================================================

def _build_transfer_matrix(leaf_ids_val, y_val, min_calibration_samples, alpha=0.0, expected_classes=None):
    """Build transfer matrix with leaf redirection for pruned leaves."""
    leaf_ids_val = np.asarray(leaf_ids_val)
    y_val = np.asarray(y_val)
    leaf_redirect = {}

    if min_calibration_samples > 0:
        counts_per_leaf = np.bincount(leaf_ids_val.astype(int))
        observed_leaves = np.unique(leaf_ids_val)
        keep_leaves = set(
            leaf for leaf in observed_leaves
            if counts_per_leaf[int(leaf)] >= min_calibration_samples
        )
        pruned_leaves = set(
            leaf for leaf in observed_leaves if leaf not in keep_leaves
        )
        keep_mask = np.array([leaf in keep_leaves for leaf in leaf_ids_val], dtype=bool)
        leaf_ids_filtered = leaf_ids_val[keep_mask]
        y_val_filtered = y_val[keep_mask]

        estimator = HoldoutTransferMatrixEstimator(alpha=alpha)
        estimator.fit(leaf_ids_filtered, y_val_filtered, classes=expected_classes)

        if len(pruned_leaves) > 0 and len(keep_leaves) > 0:
            classes = estimator.classes_
            n_classes = len(classes)
            class_to_idx = {c: i for i, c in enumerate(classes)}
            supported_leaf_list = sorted(keep_leaves)
            supported_profiles = np.zeros((len(supported_leaf_list), n_classes), dtype=float)
            for j, leaf in enumerate(supported_leaf_list):
                mask = leaf_ids_val == leaf
                for label in y_val[mask]:
                    idx = class_to_idx.get(label, None)
                    if idx is not None:
                        supported_profiles[j, idx] += 1
                total = supported_profiles[j].sum()
                if total > 0:
                    supported_profiles[j] /= total
            for leaf in pruned_leaves:
                mask = leaf_ids_val == leaf
                profile = np.zeros(n_classes, dtype=float)
                for label in y_val[mask]:
                    idx = class_to_idx.get(label, None)
                    if idx is not None:
                        profile[idx] += 1
                total = profile.sum()
                if total > 0:
                    profile /= total
                dists = np.linalg.norm(supported_profiles - profile[np.newaxis, :], axis=1)
                leaf_redirect[leaf] = supported_leaf_list[int(np.argmin(dists))]

        return estimator.P_, estimator.leaf_to_row, estimator.classes_, leaf_redirect

    estimator = HoldoutTransferMatrixEstimator(alpha=alpha)
    estimator.fit(leaf_ids_val, y_val, classes=expected_classes)
    return estimator.P_, estimator.leaf_to_row, estimator.classes_, leaf_redirect


def _counts_from_leaf_ids(leaf_ids, leaf_to_row, n_rows, leaf_redirect=None):
    """Map leaf IDs to transfer matrix rows, redirecting pruned leaves."""
    if leaf_redirect is None:
        leaf_redirect = {}
    rows = np.empty(len(leaf_ids), dtype=int)
    for i, leaf in enumerate(leaf_ids):
        row = leaf_to_row.get(leaf, -1)
        if row >= 0:
            rows[i] = row
        else:
            redirected = leaf_redirect.get(leaf, None)
            if redirected is not None:
                rows[i] = leaf_to_row.get(redirected, -1)
            else:
                rows[i] = -1
    mask = rows >= 0
    return np.bincount(rows[mask], minlength=n_rows)


def _leaf_posteriors(P, pi):
    """Compute P(class | leaf) from transfer matrix P(leaf | class) and prior pi."""
    numer = P * pi[np.newaxis, :]
    denom = numer.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return numer / denom


def _scores_from_leaf_ids(leaf_ids, leaf_to_row, leaf_post, fallback_pi):
    """Assign posterior vectors to samples based on leaf assignment."""
    scores = np.zeros((len(leaf_ids), leaf_post.shape[1]), dtype=float)
    for i, leaf_id in enumerate(leaf_ids):
        row = leaf_to_row.get(leaf_id, None)
        if row is None:
            scores[i] = fallback_pi
        else:
            scores[i] = leaf_post[row]
    return scores


# =============================================================================
# SLD (Saerens et al. EM) [2]
# =============================================================================

def _sld_em(scores, init_pi, tol=1e-6, max_iter=1000):
    """Saerens et al. EM for adjusting posteriors and estimating prevalence."""
    scores = np.asarray(scores, dtype=float)
    n_classes = scores.shape[1]
    if scores.shape[0] == 0:
        return np.ones(n_classes) / n_classes
    pi = np.asarray(init_pi, dtype=float)
    if pi.sum() <= 0:
        pi = np.ones(n_classes) / n_classes
    else:
        pi = pi / pi.sum()
    for _ in range(max_iter):
        denom = scores.dot(pi)
        denom[denom <= 0] = 1e-12
        r = (scores * pi[np.newaxis, :]) / denom[:, np.newaxis]
        new_pi = r.mean(axis=0)
        if np.linalg.norm(new_pi - pi, ord=1) < tol:
            pi = new_pi
            break
        pi = new_pi
    pi = np.maximum(pi, 0)
    if pi.sum() == 0:
        return np.ones(n_classes) / n_classes
    return pi / pi.sum()


# =============================================================================
# Calibration helpers
# =============================================================================

def _fit_platt_calibrators(probs_val, y_val, classes):
    """Fit per-class Platt calibrators on validation data. Returns list of fitted calibrators."""
    probs_val = np.asarray(probs_val, dtype=float)
    y_val = np.asarray(y_val)
    classes = np.asarray(classes)
    calibrators = []
    for idx, cls in enumerate(classes):
        y_bin = (y_val == cls).astype(int)
        if np.unique(y_bin).size < 2:
            calibrators.append(None)
            continue
        cal = LogisticRegression(max_iter=1000, solver="lbfgs")
        cal.fit(probs_val[:, [idx]], y_bin)
        calibrators.append(cal)
    return calibrators


def _apply_calibrators(calibrators, probs_test, classes):
    """Apply pre-fitted calibrators to test probabilities."""
    probs_test = np.asarray(probs_test, dtype=float)
    calibrated = np.zeros_like(probs_test, dtype=float)
    for idx, cal in enumerate(calibrators):
        if cal is None:
            calibrated[:, idx] = probs_test[:, idx]
        else:
            calibrated[:, idx] = cal.predict_proba(probs_test[:, [idx]])[:, 1]
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return calibrated / row_sums


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


# =============================================================================
# Pre-fitted estimator wrappers (fit once, quantify once per test set)
# =============================================================================

class _FittedEMEstimator:
    """Pre-fitted EM/ACC estimator."""

    def __init__(self, tree, P, leaf_to_row, leaf_redirect, classes, solver):
        self.tree = tree
        self.P = P
        self.leaf_to_row = leaf_to_row
        self.leaf_redirect = leaf_redirect
        self.classes = classes
        self.solver = solver

    def quantify(self, X_test):
        leaf_ids = self.tree.get_leaf_indices(X_test)
        counts = _counts_from_leaf_ids(
            leaf_ids, self.leaf_to_row, self.P.shape[0],
            leaf_redirect=self.leaf_redirect
        )
        try:
            return self.solver.estimate_prevalence(counts, self.P, init_pi=None)
        except TypeError:
            return self.solver.estimate_prevalence(counts, self.P)


class _FittedSLDEstimator:
    """Pre-fitted SLD estimator."""

    def __init__(self, tree, P, leaf_to_row, classes, pi_val, pi_train):
        self.tree = tree
        self.leaf_to_row = leaf_to_row
        self.classes = classes
        self.pi_val = pi_val
        self.pi_train = pi_train
        self.leaf_post = _leaf_posteriors(P, pi_val)

    def quantify(self, X_test):
        leaf_ids = self.tree.get_leaf_indices(X_test)
        scores = _scores_from_leaf_ids(
            leaf_ids, self.leaf_to_row, self.leaf_post, self.pi_val
        )
        return _sld_em(scores, init_pi=self.pi_train)


class _FittedPlattSLDEstimator:
    """Pre-fitted Platt-calibrated SLD estimator."""

    def __init__(self, base_classifier, calibrators, classes, pi_train):
        self.base_classifier = base_classifier
        self.calibrators = calibrators
        self.classes = classes
        self.pi_train = pi_train

    def quantify(self, X_test):
        probs_test = _align_probabilities_to_classes(
            self.base_classifier.predict_proba(X_test),
            self.base_classifier.classes_, self.classes
        )
        scores_test = _apply_calibrators(self.calibrators, probs_test, self.classes)
        return _sld_em(scores_test, init_pi=self.pi_train)


class _FittedLRPlattSLDEstimator:
    """Pre-fitted LR + Platt-calibrated SLD estimator."""

    def __init__(self, lr_model, calibrators, classes, pi_train):
        self.lr_model = lr_model
        self.calibrators = calibrators
        self.classes = classes
        self.pi_train = pi_train

    def quantify(self, X_test):
        probs_test = _align_probabilities_to_classes(
            self.lr_model.predict_proba(X_test),
            self.lr_model.classes_, self.classes
        )
        scores_test = _apply_calibrators(self.calibrators, probs_test, self.classes)
        return _sld_em(scores_test, init_pi=self.pi_train)


class _FittedKDEyEstimator:
    """Pre-fitted KDEyML wrapper."""

    def __init__(self, kdey_model, classes):
        self.kdey_model = kdey_model
        self.classes = classes

    def quantify(self, X_test):
        return self.kdey_model.quantify(X_test)


# =============================================================================
# Stratified train/val split from training data
# =============================================================================

def _split_train_val(X_train, y_train, val_fraction, seed):
    """
    Split training data into train and validation (stratified).
    Validation is used for transfer matrix estimation / calibration.
    """
    from sklearn.model_selection import train_test_split
    if val_fraction <= 0 or val_fraction >= 1.0:
        return X_train, y_train, X_train, y_train
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=val_fraction,
            stratify=y_train, random_state=seed
        )
    except ValueError:
        # Stratification fails with too few samples per class — fallback
        X_tr, y_tr = X_train, y_train
        X_val, y_val = X_train, y_train
    return X_tr, y_tr, X_val, y_val


# =============================================================================
# Run a single work unit: one (dataset, seed, dt_ratio, train_distr, test_distr)
# =============================================================================

def run_single_unit(
    dta_name, seed, dt_ratio, train_distr, test_distr,
    trees, estimators, min_calibration_samples,
    max_depth, min_samples_leaf, kde_bandwidth,
    qeb_max_features, qeb_max_thresholds,
    load_from_disk, em_alpha, verbose,
):
    """
    Run all tree/estimator/min_cal combinations for a single
    (dataset, seed, dt_ratio, train_distr, test_distr) draw.

    This follows the paper's protocol [2]:
    - One synthetic draw per combination
    - Fit quantifier on training portion
    - Predict on test portion
    - Repeat across seeds for variance [2]
    """
    X, y, N, Y, n_classes, y_cts, y_idx = helpers.get_xy(
        dta_name, load_from_disk=load_from_disk, binned=False,
    )

    # Map labels to integer indices
    class_list = np.arange(n_classes)
    class_to_idx = {c: i for i, c in enumerate(Y)}

    def _map_labels(arr):
        return np.array([class_to_idx.get(v, v) for v in arr])

    # Synthetic draw — one sample per combination [1][2]
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

    # Stratified split of training into train + validation (75/25)
    X_tr, y_tr, X_val, y_val = _split_train_val(
        X_train_full, y_train_full, val_fraction=0.25, seed=seed
    )

    # True test prevalence
    pi_true = _prevalence(y_test, class_list)

    rows = []

    for tree_name in trees:
        tree = _build_tree(
            tree_name, max_depth, min_samples_leaf, seed,
            qeb_max_features=qeb_max_features,
            qeb_max_thresholds=qeb_max_thresholds,
        )

        # Fit tree ONCE on training data
        tree.fit(X_tr, y_tr)
        leaf_ids_val = tree.get_leaf_indices(X_val)

        for min_cal in min_calibration_samples:
            # Cache transfer matrices by alpha
            P_cache = {}

            def get_P(alpha, _min_cal=min_cal):
                key = float(alpha)
                if key not in P_cache:
                    P_cache[key] = _build_transfer_matrix(
                        leaf_ids_val, y_val, _min_cal, alpha=alpha,
                        expected_classes=class_list,
                    )
                return P_cache[key]

            for est_name in estimators:
                if verbose:
                    print(
                        f"  dt={dt_ratio}, train_d={train_distr}, "
                        f"test_d={test_distr}, tree={tree_name}, "
                        f"est={est_name}, min_cal={min_cal}"
                    )

                # ==============================================================
                # FIT PHASE — done ONCE per (tree, min_cal, estimator)
                # ==============================================================

                if est_name in {"em", "em_smooth"}:
                    alpha = em_alpha if est_name == "em_smooth" else 0.0
                    P, leaf_to_row, classes, leaf_redirect = get_P(alpha)
                    fitted = _FittedEMEstimator(
                        tree, P, leaf_to_row, leaf_redirect, classes,
                        EMQuantificationSolver()
                    )

                elif est_name == "acc":
                    P, leaf_to_row, classes, leaf_redirect = get_P(0.0)
                    fitted = _FittedEMEstimator(
                        tree, P, leaf_to_row, leaf_redirect, classes,
                        ACCQuantificationSolver()
                    )

                elif est_name == "sld":
                    P, leaf_to_row, classes, _ = get_P(0.0)
                    pi_val = _prevalence(y_val, classes)
                    pi_train = _prevalence(y_tr, classes)
                    fitted = _FittedSLDEstimator(
                        tree, P, leaf_to_row, classes, pi_val, pi_train
                    )

                elif est_name == "platt_sld":
                    P, leaf_to_row, classes, _ = get_P(0.0)
                    base = TreePosteriorClassifier(
                        tree=tree, alpha=0.01,
                        min_calibration_samples=min_cal,
                        expected_classes=classes,
                    )
                    base.fit(X_tr, y_tr)
                    probs_val = _align_probabilities_to_classes(
                        base.predict_proba(X_val), base.classes_, classes
                    )
                    calibrators = _fit_platt_calibrators(probs_val, y_val, classes)
                    pi_train = _prevalence(y_tr, classes)
                    fitted = _FittedPlattSLDEstimator(
                        base, calibrators, classes, pi_train
                    )

                elif est_name == "lr_platt_sld":
                    P, leaf_to_row, classes, _ = get_P(0.0)
                    lr_model = LogisticRegression(max_iter=1000, solver="lbfgs")
                    lr_model.fit(X_tr, y_tr)
                    probs_val = _align_probabilities_to_classes(
                        lr_model.predict_proba(X_val), lr_model.classes_, classes
                    )
                    calibrators = _fit_platt_calibrators(probs_val, y_val, classes)
                    pi_train = _prevalence(y_tr, classes)
                    fitted = _FittedLRPlattSLDEstimator(
                        lr_model, calibrators, classes, pi_train
                    )

                elif est_name == "kdey":
                    if not QUAPY_AVAILABLE:
                        continue
                    P, leaf_to_row, classes, _ = get_P(0.0)
                    wrapper = TreePosteriorClassifier(
                        tree=tree, alpha=0.01,
                        min_calibration_samples=min_cal,
                        expected_classes=classes,
                    )
                    kdey = KDEyML(
                        classifier=wrapper,
                        fit_classifier=True,
                        val_split=(X_val, y_val),
                        bandwidth=kde_bandwidth,
                        random_state=seed,
                    )
                    kdey.fit(X_tr, y_tr)
                    fitted = _FittedKDEyEstimator(kdey, classes)

                else:
                    raise ValueError(f"Unknown estimator: {est_name}")

                # ==============================================================
                # PREDICT PHASE — single prediction on test set
                # ==============================================================

                pi_hat = fitted.quantify(X_test)
                pi_hat = np.asarray(pi_hat, dtype=float)

                # Ensure alignment
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
    # MODE 1: Manifest-based parallelization (round-robin across tasks)
    # ------------------------------------------------------------------
    if args.manifest is not None:
        assert args.task_id is not None and args.n_tasks is not None, \
            "--task-id and --n-tasks are required when using --manifest"

        with open(args.manifest) as f:
            all_units = [line.strip().split(",") for line in f if line.strip()]

        # Round-robin: each task gets every N-th unit (mixes large/small datasets)
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
                dta_name=dta_name,
                seed=seed,
                dt_ratio=dt_ratio,
                train_distr=train_distr,
                test_distr=test_distr,
                trees=args.trees,
                estimators=args.estimators,
                min_calibration_samples=args.min_calibration_samples,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                kde_bandwidth=args.kde_bandwidth,
                qeb_max_features=args.qeb_max_features,
                qeb_max_thresholds=args.qeb_max_thresholds,
                load_from_disk=args.load_from_disk,
                em_alpha=args.em_alpha,
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
    # MODE 2: Standard execution (no manifest — runs all combinations)
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

    # Determine train/test ratios following the paper [2]:
    # (0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3)
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
                            dta_name=dta_name,
                            seed=seed,
                            dt_ratio=dt_ratio,
                            train_distr=train_distr,
                            test_distr=test_distr,
                            trees=args.trees,
                            estimators=args.estimators,
                            min_calibration_samples=args.min_calibration_samples,
                            max_depth=args.max_depth,
                            min_samples_leaf=args.min_samples_leaf,
                            kde_bandwidth=args.kde_bandwidth,
                            qeb_max_features=args.qeb_max_features,
                            qeb_max_thresholds=args.qeb_max_thresholds,
                            load_from_disk=args.load_from_disk,
                            em_alpha=args.em_alpha,
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