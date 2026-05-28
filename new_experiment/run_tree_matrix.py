import argparse
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import quapy as qp
from quapy.method.aggregative import KDEyML, EMQ, ACC, PACC
from quapy.data import LabelledCollection
from quapy.protocol import APP
from sklearn.linear_model import LogisticRegression

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
)
from new_experiment.impl import (
    ClassificationTree,
    QuantificationErrorBalancingTree,
    ClassificationQuantificationBalancingTree,
    HoldoutTransferMatrixEstimator,
    EMQuantificationSolver,
    ACCQuantificationSolver,
)

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from new_experiment.impl import (
    ClassificationTree,
    QuantificationErrorBalancingTree,
    ClassificationQuantificationBalancingTree,
    HoldoutTransferMatrixEstimator,
)


class TreePosteriorClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, tree=None, alpha=0.01, min_calibration_samples=0):
        self.tree = tree
        self.alpha = alpha
        self.min_calibration_samples = min_calibration_samples

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.tree.fit(X, y)
        leaf_ids = self.tree.get_leaf_indices(X)

        # Apply min_calibration_samples pruning
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
                # Build redirect map
                n_classes = len(self.classes_)
                class_to_idx = {c: i for i, c in enumerate(self.classes_)}

                supported_leaf_list = sorted(keep_leaves)
                supported_profiles = np.zeros(
                    (len(supported_leaf_list), n_classes), dtype=float
                )
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
                    dists = np.linalg.norm(
                        supported_profiles - profile[np.newaxis, :], axis=1
                    )
                    self.leaf_redirect_[leaf] = supported_leaf_list[int(np.argmin(dists))]

            # Filter for transfer matrix estimation
            keep_mask = np.array(
                [leaf in keep_leaves for leaf in leaf_ids_filtered], dtype=bool
            )
            leaf_ids_filtered = leaf_ids_filtered[keep_mask]
            y_filtered = y_filtered[keep_mask]

        # Build transfer matrix from (possibly pruned) data
        estimator = HoldoutTransferMatrixEstimator(alpha=self.alpha)
        estimator.fit(leaf_ids_filtered, y_filtered)
        self.P_ = estimator.P_
        self.leaf_to_row_ = estimator.leaf_to_row

        # Compute posteriors P(Y|leaf)
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
                # Try redirect
                redirected = self.leaf_redirect_.get(lid, None)
                if redirected is not None:
                    row = self.leaf_to_row_.get(redirected, None)
            if row is not None:
                proba[i] = self.leaf_posteriors_[row]

        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

TREE_CHOICES = ["gini", "qeb", "qcqb"]
ESTIMATOR_CHOICES = ["em", "em_smooth", "acc", "sld", "kdey"]


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
    sample_size = len(p_true)
    return {
        "AE": _ae(p_true, p_hat),
        "RAE": _rae(p_true, p_hat, eps=_calc_eps(sample_size)),
        "NKLD": _nkld(p_true, p_hat, eps=_calc_eps(sample_size)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--datasets", nargs="*", default=DATASET_LIST,
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
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
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
    return parser.parse_args()


def _split_train_val_test(X, y, train_fraction, val_fraction, seed):
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_fraction, stratify=y, random_state=seed
    )
    val_size = val_fraction / (1.0 - train_fraction)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_size, stratify=y_temp, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


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


def _leaf_posteriors(P, pi):
    """Compute P(class | leaf) from transfer matrix P(leaf | class) and prior pi."""
    numer = P * pi[np.newaxis, :]
    denom = numer.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return numer / denom


def _scores_from_leaf_ids(leaf_ids, leaf_to_row, leaf_post, fallback_pi):
    """Assign posterior probability vectors to samples based on their leaf assignment."""
    scores = np.zeros((len(leaf_ids), leaf_post.shape[1]), dtype=float)
    for i, leaf_id in enumerate(leaf_ids):
        row = leaf_to_row.get(leaf_id, None)
        if row is None:
            scores[i] = fallback_pi
        else:
            scores[i] = leaf_post[row]
    return scores


def _counts_from_leaf_ids(leaf_ids, leaf_to_row, n_rows):
    rows = np.array([leaf_to_row.get(l, -1) for l in leaf_ids], dtype=int)
    mask = rows >= 0
    return np.bincount(rows[mask], minlength=n_rows)


def _build_transfer_matrix(leaf_ids_val, y_val, min_calibration_samples, alpha=0.0):
    """Build transfer matrix with leaf redirection for pruned leaves.
    
    Returns (P, leaf_to_row, classes, leaf_redirect) where leaf_redirect
    maps unsupported leaf IDs to supported ones.
    """
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

        # Filter to supported leaves for estimator fitting
        keep_mask = np.array([leaf in keep_leaves for leaf in leaf_ids_val], dtype=bool)
        leaf_ids_filtered = leaf_ids_val[keep_mask]
        y_val_filtered = y_val[keep_mask]

        estimator = HoldoutTransferMatrixEstimator(alpha=alpha)
        estimator.fit(leaf_ids_filtered, y_val_filtered)

        # Build redirect map for pruned leaves
        if len(pruned_leaves) > 0 and len(keep_leaves) > 0:
            classes = estimator.classes_
            n_classes = len(classes)
            class_to_idx = {c: i for i, c in enumerate(classes)}

            # Compute profiles for supported leaves
            supported_leaf_list = sorted(keep_leaves)
            supported_profiles = np.zeros((len(supported_leaf_list), n_classes), dtype=float)
            for j, leaf in enumerate(supported_leaf_list):
                mask = leaf_ids_val == leaf
                y_leaf = y_val[mask]
                for label in y_leaf:
                    idx = class_to_idx.get(label, None)
                    if idx is not None:
                        supported_profiles[j, idx] += 1
                total = supported_profiles[j].sum()
                if total > 0:
                    supported_profiles[j] /= total

            # Redirect each pruned leaf to closest supported leaf
            for leaf in pruned_leaves:
                mask = leaf_ids_val == leaf
                y_leaf = y_val[mask]
                profile = np.zeros(n_classes, dtype=float)
                for label in y_leaf:
                    idx = class_to_idx.get(label, None)
                    if idx is not None:
                        profile[idx] += 1
                total = profile.sum()
                if total > 0:
                    profile /= total
                dists = np.linalg.norm(supported_profiles - profile[np.newaxis, :], axis=1)
                closest_idx = int(np.argmin(dists))
                leaf_redirect[leaf] = supported_leaf_list[closest_idx]

        return estimator.P_, estimator.leaf_to_row, estimator.classes_, leaf_redirect

    estimator = HoldoutTransferMatrixEstimator(alpha=alpha)
    estimator.fit(leaf_ids_val, y_val)
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


def _kdey_quantify_quapy(
    X_train, y_train, X_val, y_val, X_test, y_test,
    tree, leaf_ids_val, leaf_ids_test, P, leaf_to_row, classes,
    bandwidth, seed, min_calibration_samples
):
    """
    Use QuaPy's KDEyML to perform quantification.
    
    KDEyML uses k-fold cross-validation to generate training posteriors,
    fits per-class KDEs on the multivariate posterior vectors on the simplex,
    and maximizes the log-likelihood of test posteriors under the 
    class-conditional KDE mixture [1].
    """
    from quapy.method.aggregative import KDEyML
    from sklearn.linear_model import LogisticRegression

    wrapper = TreePosteriorClassifier(tree=tree, alpha=0.01, min_calibration_samples=min_calibration_samples)
        #max_iter=1000,
       # random_state=seed,
    #)

    kdey = KDEyML(
        classifier=wrapper,
        fit_classifier=True,
        val_split=(X_val,y_val),       # 5-fold CV for training posteriors [1]
        bandwidth=bandwidth,
        random_state=seed,
    )

    # Use the separate X, y API instead of LabelledCollection
    kdey.fit(X_train, y_train)

    # Quantify the test set
    pi_hat = kdey.quantify(X_test)

    return pi_hat

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


def run_dataset(
    dta_name, seed, trees, estimators, min_calibration_samples,
    train_fraction, val_fraction, max_depth, min_samples_leaf,
    kde_bandwidth, verbose=False, qeb_max_features=None,
    qeb_max_thresholds=None, load_from_disk=False, em_alpha=1.0,
):
    X, y, _, Y, n_classes, _, _ = helpers.get_xy(
        dta_name, load_from_disk=load_from_disk, binned=False,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = _split_train_val_test(
        X, y, train_fraction, val_fraction, seed
    )
    pi_true = _prevalence(y_test, Y)

    rows = []
    total_configs = len(trees) * len(min_calibration_samples) * len(estimators)
    config_index = 0

    for tree_name in trees:
        tree = _build_tree(
            tree_name, max_depth, min_samples_leaf, seed,
            qeb_max_features=qeb_max_features,
            qeb_max_thresholds=qeb_max_thresholds,
        )
        # Ensure labels are discrete integer indices expected by sklearn
        class_list = np.array(Y)
        class_to_idx = {c: i for i, c in enumerate(class_list)}

        def _map_labels(arr):
            return np.array([class_to_idx.get(v, v) for v in arr])

        y_train = _map_labels(y_train)
        y_val = _map_labels(y_val)
        y_test = _map_labels(y_test)

        # Recompute true prevalence to match mapped labels
        pi_true = _prevalence(y_test, np.arange(len(class_list)))

        tree.fit(X_train, y_train)
        leaf_ids_val = tree.get_leaf_indices(X_val)
        leaf_ids_test = tree.get_leaf_indices(X_test)

        for min_cal in min_calibration_samples:
            P_cache = {}

            def get_P(alpha):
                key = float(alpha)
                if key not in P_cache:
                    P_cache[key] = _build_transfer_matrix(
                        leaf_ids_val, y_val, min_cal, alpha=alpha,
                    )
                return P_cache[key]

            for est_name in estimators:
                config_index += 1
                if verbose:
                    print(
                        f"  Config {config_index}/{total_configs}: tree={tree_name}, "
                        f"estimator={est_name}, min_cal={min_cal}"
                    )

                if est_name in {"em", "em_smooth", "acc"}:
                    solver = (
                        EMQuantificationSolver()
                        if est_name in {"em", "em_smooth"}
                        else ACCQuantificationSolver()
                    )
                    alpha = em_alpha if est_name == "em_smooth" else 0.0
                    P, leaf_to_row, classes, leaf_redirect = get_P(alpha)
                    pi_true_local = _prevalence(y_test, classes)
                    counts = _counts_from_leaf_ids(
                        leaf_ids_test, leaf_to_row, P.shape[0],
                        leaf_redirect=leaf_redirect
                    )
                    try:
                        pi_hat = solver.estimate_prevalence(counts, P, init_pi=None)
                    except TypeError:
                        pi_hat = solver.estimate_prevalence(counts, P)

                elif est_name == "sld":
                    # SLD (EMQ / Saerens et al.) using leaf posteriors as scores
                    P, leaf_to_row, classes, _ = get_P(0.0)
                    pi_val = _prevalence(y_val, classes)
                    pi_train = _prevalence(y_train, classes)
                    pi_true_local = _prevalence(y_test, classes)
                    leaf_post = _leaf_posteriors(P, pi_val)
                    scores_test = _scores_from_leaf_ids(
                        leaf_ids_test, leaf_to_row, leaf_post, pi_val
                    )
                    pi_hat = _sld_em(scores_test, init_pi=pi_train)

                elif est_name == "kdey":
                    # =====================================================
                    # USE QUAPY's KDEyML — PROPER IMPLEMENTATION
                    # =====================================================
                    # This uses QuaPy's implementation which:
                    # 1. Trains a Logistic Regression classifier [1]
                    # 2. Generates cross-validated posteriors for training
                    #    samples (avoiding data leakage) [1]
                    # 3. Fits per-class KDEs on MULTIVARIATE posterior
                    #    vectors on the simplex [1]
                    # 4. Maximizes log-likelihood of test posteriors under
                    #    the class-conditional KDE mixture [1]
                    # =====================================================
                    P, leaf_to_row, classes, _  = get_P(0.0)
                    pi_hat = _kdey_quantify_quapy(
                        X_train, y_train, X_val, y_val, X_test, y_test,
                        tree, leaf_ids_val, leaf_ids_test,
                        P, leaf_to_row, classes,
                        bandwidth=kde_bandwidth,
                        seed=seed,
                        min_calibration_samples=min_cal,
                    )
                    classes = Y
                    pi_true_local = _prevalence(y_test, classes)

                else:
                    raise ValueError(f"Unknown estimator: {est_name}")

                metrics = _evaluate(pi_true_local, pi_hat)
                rows.append({
                    "dataset": dta_name,
                    "seed": seed,
                    "n_classes": n_classes,
                    "tree": tree_name,
                    "estimator": est_name,
                    "min_calibration_samples": int(min_cal),
                    "AE": metrics["AE"],
                    "RAE": metrics["RAE"],
                    "NKLD": metrics["NKLD"],
                    "true_prevalence": ";".join(str(v) for v in pi_true_local),
                    "pred_prevalence": ";".join(str(v) for v in pi_hat),
                })

    return rows


def main():
    args = parse_args()
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

    all_rows = []
    total_runs = len(datasets) * len(args.seeds)
    run_index = 0

    for dta_name in datasets:
        for seed in args.seeds:
            run_index += 1
            print(f"Running {dta_name} (seed={seed}) [{run_index}/{total_runs}]")
            rows = run_dataset(
                dta_name=dta_name,
                seed=seed,
                trees=args.trees,
                estimators=args.estimators,
                min_calibration_samples=args.min_calibration_samples,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                kde_bandwidth=args.kde_bandwidth,
                verbose=args.verbose,
                qeb_max_features=args.qeb_max_features,
                qeb_max_thresholds=args.qeb_max_thresholds,
                load_from_disk=args.load_from_disk,
                em_alpha=args.em_alpha,
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
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()