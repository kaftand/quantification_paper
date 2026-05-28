# Revised Specification: Decision Tree-Based Quantification

---

## Abstract Classes

---

[Abstract Class]
**QuantificationTree**

[Properties]
- `tree_`: a fitted scikit-learn tree (DecisionTreeClassifier or DecisionTreeRegressor depending on criterion)
- `n_classes_`: number of classes (K)
- `classes_`: array of class labels
- `n_leaves_`: number of leaves in the fitted tree
- `min_leaf_size`: minimum number of instances in a leaf [1]
- `max_depth`: maximum tree depth

[Methods]
- `fit(X_train, y_train)`: build the tree from training data using the type-specific fitting strategy
- `get_leaf_indices(X)`: return leaf index for each instance via `tree_.apply(X)`

---

[Abstract Class]
**TransferMatrixEstimator**

[Properties]
- `n_leaves_`: number of bins (leaves)
- `n_classes_`: number of classes
- `P_`: transfer matrix of shape (n_leaves_, n_classes_), where P[j, k] = P(leaf_j | Y=k)

[Methods]
- `fit(leaf_indices, y)`: estimate P by counting co-occurrences on held-out data
- `get_matrix()`: return P

---

[Abstract Class]
**QuantificationSolver**

[Properties]
- `tol`: convergence tolerance
- `max_iter`: maximum iterations

[Methods]
- `estimate_prevalence(counts, P, init_pi=None)`: return estimated class prevalences

---

[Abstract Class]
**Quantifier**

[Properties]
- `tree`: a QuantificationTree instance
- `matrix_estimator`: a TransferMatrixEstimator
- `solver`: a QuantificationSolver
- `classes_`: class labels

[Methods]
- `fit(X_train, y_train, X_val, y_val)`: fit tree on training, estimate transfer matrix on validation
- `quantify(X_test)`: return prevalence estimates

---

## Tree Types (Defined by Fitting Criteria)

---

[Method]
**ClassificationTree** — Wraps a scikit-learn `DecisionTreeClassifier` with `criterion='gini'` (or `'entropy'`). This is what Börner et al. [1] implicitly use: "a tree to classify the events in the different energy bins" with cuts "optimized to separate the data as good as possible" [1]. The tree is trained to predict the class label, and leaves are used purely as a partition of feature space.

[Test]
Construct a dataset with 4 features where only feature 0 perfectly separates two classes (class 0 has feature_0 < 0.5, class 1 has feature_0 ≥ 0.5, other features are random noise). Fit a ClassificationTree with max_depth=1. Verify the single split is on feature 0. Verify the resulting two leaves partition all instances (every instance gets exactly one leaf index). Verify `n_leaves_ == 2`.

---

[Method]
**QuantificationErrorBalancingTree (QEB)** — Wraps a tree built using the Quantification Error Balancing criterion from Milli et al. [4]. For each candidate split, computes $E_{c_i} = |FP_{c_i} - FN_{c_i}|$ per class. The gain is $\Delta = \|QE^{parent}\|_2 - \|QE^{child}\|_2$. Splits are accepted only if $\Delta > 0$ [4]. Leaves are assigned majority class labels internally (for FP/FN computation during tree building), but for quantification purposes we only use the leaf indices as bin assignments.

**Note:** Since this cannot use scikit-learn's built-in split criteria, it requires a custom tree-building implementation OR a pre-built tree passed in. For pragmatic implementation, one option is to use the quantification forest code from Milli et al. [4][5] and extract leaf assignments.

A sample implementation can be found in QFY\classification_models\_qforest.py, however this is for a forest, not a tree.

[Test]
Construct a binary dataset (100 instances: 60 class 0, 40 class 1) with two features. Feature A creates a split with |FP - FN| = 15 (unbalanced errors), feature B creates a split with |FP - FN| = 0 (perfectly balanced errors). Verify QEB selects feature B. Then fit a ClassificationTree (Gini) on the same data and verify it selects feature A. This demonstrates the two criteria produce different trees on the same data.

---

[Method]
**ClassificationQuantificationBalancingTree (QCQB)** — Wraps a tree built using the Classification-Quantification Balancing criterion from Milli et al. [4]. For each class $c_i$, computes $E_{c_i} = |FP_{c_i}^2 - FN_{c_i}^2| = |FP_{c_i} - FN_{c_i}| \times |FP_{c_i} + FN_{c_i}|$ [4]. The gain is $\Delta = \|QE^{parent}\|_2 - \|QE^{child}\|_2$ [4]. This balances quantification accuracy with classification accuracy.

[Test]
Construct a 3-class dataset where two candidate splits both achieve |FP - FN| = 0 for all classes (both are perfect quantifiers), but split A has total misclassifications FP + FN = 40 and split B has FP + FN = 10. Verify QCQB prefers split B (lower $\|QE\|_2$ due to the squared term penalizing total errors). Verify QEB is indifferent between them (both have $\Delta = 0$ since parent already has |FP - FN| = 0 after the split). This demonstrates QCQB's classification-quantification trade-off [4].

---

[Method]
**ConditionNumberTree** — A tree built to minimize the condition number $\kappa$ of the resulting transfer matrix $M$, as motivated by Börner et al. [1]: "The higher the ambiguity of the measurement the higher is the condition number $\kappa$ of matrix $A$ and the more ill-posed is the problem" [1]. At each node, candidate splits are evaluated by tentatively computing the transfer matrix implied by the resulting leaf structure and selecting the split that yields the lowest $\kappa(M)$.

**Note:** This requires computing the transfer matrix at each candidate split (expensive), so a validation set or the training set itself is used for provisional column estimates during tree construction.

[Test]
Construct a 3-class dataset (150 instances, 50 per class) with two features. Feature A perfectly separates class 0 from {1, 2} but leaves classes 1 and 2 completely mixed (yielding a transfer matrix with two nearly identical columns, high $\kappa$). Feature B partially separates all three classes (yielding a transfer matrix with distinct columns, lower $\kappa$). Fit a ConditionNumberTree with max_depth=1. Verify it selects feature B. Compute the condition number of the resulting 2×3 transfer matrix and verify it is lower than what feature A's split would produce. Fit a ClassificationTree on the same data and verify it selects feature A (since separating one class perfectly maximizes Gini gain). This demonstrates that optimizing for $\kappa$ produces different partitions than optimizing for classification purity [1].

---

[Method]
**RandomForestTree** — An ensemble of ClassificationTrees built via bagging (bootstrap samples + random feature subsets), following the standard Random Forest approach [4][5]. Each tree contributes its own set of leaves. For quantification, there are two modes of use: (a) each tree independently produces a transfer matrix and prevalence estimate, and estimates are averaged; or (b) all trees' leaf systems are concatenated into one large transfer matrix and solved once. This is analogous to the random forest approach in [4] but using leaves as bins rather than as classifiers.

[Test]
**Test 1 (multiple trees):** Fit a RandomForestTree ensemble of 50 trees on 500 training instances (3 classes). Verify that each tree has a different leaf structure (different number of leaves or different leaf assignments for the same instance). Verify every instance is assigned to exactly one leaf per tree.

**Test 2 (variance reduction):** Generate 30 test bags of 500 instances each from the same shifted distribution. Run a single ClassificationTree quantifier and a RandomForestTree quantifier (50 trees, mode (a): averaged estimates) on each bag. Verify the variance of prevalence estimates across bags is lower for the forest than for the single tree.

---

## Methods for Quantification

---

[Method]
**HoldoutTransferMatrixEstimator.fit(leaf_indices, y)** — Estimates P[j, k] = (# val instances of class k in leaf j) / (# val instances of class k). Each column sums to 1.

[Test]
Create 300 validation instances: 100 per class (3 classes). Manually assign leaf indices such that class 0 sends 80 to leaf 0, 10 to leaf 1, 10 to leaf 2; class 1 sends 10, 80, 10; class 2 sends 10, 10, 80. Verify fitted P equals [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]. Verify each column sums to 1.0. Verify shape is (3, 3).

---

[Method]
**EMQuantificationSolver.estimate_prevalence(counts, P, init_pi, tol, max_iter)** — EM/IBU solver [2]. Iterates E-step (compute responsibilities) and M-step (update prevalences from weighted counts) until convergence.

[Test]
**Test 1 (exact recovery under no shift):** P = [[0.9, 0.1], [0.1, 0.9]], true π* = [0.5, 0.5]. Expected counts = [500, 500] for N=1000. Verify output is [0.5, 0.5] within 1e-6.

**Test 2 (recovery under shift):** Same P. True π* = [0.3, 0.7]. Expected counts = [340, 660]. Init with [0.5, 0.5]. Verify output converges to [0.3, 0.7] within 1e-4.

---

[Method]
**ACCQuantificationSolver.estimate_prevalence(counts, P)** — Solve q = P @ π for π via constrained least squares: minimize ||q - P @ π||² subject to π ≥ 0, Σπ = 1 [2].

[Test]
P = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]. True π* = [0.5, 0.3, 0.2]. Compute q = P @ π*. Verify output is [0.5, 0.3, 0.2] within 1e-6.

---

[Method]
**TreeBinQuantifier.fit(X_train, y_train, X_val, y_val)** — (1) Fit the selected QuantificationTree type on X_train, y_train. (2) Compute leaf_indices = tree.get_leaf_indices(X_val). (3) Fit transfer matrix on (leaf_indices, y_val).

[Test]
Fit using a ClassificationTree on 500 training instances (3 classes, 10 features) and 200 validation instances. Verify tree is fitted, P has shape (n_leaves_, 3), each column of P sums to 1.0, no column is all zeros.

---

[Method]
**TreeBinQuantifier.quantify(X_test)** — Map test instances to leaves, count leaf occurrences, call solver.

[Test]
**Test 1 (shift recovery):** Train with π = [0.33, 0.33, 0.34]. Generate test set of 1000 instances with true π* = [0.7, 0.2, 0.1] by resampling class-conditionally. Verify L1 error < 0.1.

**Test 2 (calibration independence):** After fitting, verify that `quantify` produces identical output regardless of what `tree.tree_.predict_proba` would return — because we only use `tree.get_leaf_indices` (i.e., `tree_.apply`). Concretely: corrupt the tree's internal class probability arrays at each leaf, re-run quantify, and verify the result is unchanged.

---

## Evaluation Matrix (Experimental Comparison)

For benchmarking, combine each tree type with each solver:

| | **EM Solver** | **ACC Solver** |
|---|---|---|
| **ClassificationTree (Gini)** [1] | CT+EM | CT+ACC |
| **QEB Tree** [4] | QEB+EM | QEB+ACC (= QF-AC from [4][5]) |
| **QCQB Tree** [4] | QCQB+EM | QCQB+ACC |
| **ConditionNumberTree** [1] | κ-Tree+EM | κ-Tree+ACC |
| **RandomForestTree** | RF+EM | RF+ACC |

Compare against baselines from Schumacher et al. [5]: SLD/EM (with logistic regression posteriors), ACC (standard), HDy, MS, FMM, HDx, GPAC.

---

## Implementation

A minimal reference implementation accompanying these specs is provided under the `new_experiment/impl/` subdirectory. It implements the core components required to run the experiments described above:

- `tree.py`: a thin `ClassificationTree` wrapper around `sklearn.tree.DecisionTreeClassifier`.
- `estimators.py`: `HoldoutTransferMatrixEstimator` that fits P from validation leaf labels.
- `solvers.py`: `EMQuantificationSolver` (EM/IBU) and `ACCQuantificationSolver` (least-squares + simplex projection).
- `quantifier.py`: `TreeBinQuantifier` that wires tree + estimator + solver to provide `fit`/`quantify`.

See `new_experiment/impl/README.md` for quick run instructions using the `quant` conda environment.

---

## Implemented components

The reference implementation now includes additional tree variants described in the specification. These are implemented as readable, greedy reference algorithms intended for experimentation and testing (not optimized for production):

- `QuantificationErrorBalancingTree`: greedy binary tree that evaluates candidate splits by the reduction in quantification-error imbalance (QEB). Implemented in [new_experiment/impl/tree_variants.py](new_experiment/impl/tree_variants.py#L1).
- `ClassificationQuantificationBalancingTree`: variant that uses the squared FP/FN formulation (QCQB) to balance classification and quantification objectives. Implemented in [new_experiment/impl/tree_variants.py](new_experiment/impl/tree_variants.py#L1).
- `ConditionNumberTree`: greedy tree that selects splits minimizing the condition number of the provisional transfer matrix computed on a validation set. Implemented in [new_experiment/impl/tree_variants.py](new_experiment/impl/tree_variants.py#L1).
- `RandomForestTree`: ensemble wrapper that builds multiple `ClassificationTree` instances via bootstrap and supports two aggregation modes: `per_tree` (average per-tree prevalence estimates) and `concat` (concatenate all trees' leaf systems and solve once). Implemented in [new_experiment/impl/tree_variants.py](new_experiment/impl/tree_variants.py#L1).

Notes and limitations:
- The QEB/QCQB/ConditionNumber trees use a straightforward greedy splitter that evaluates thresholds at midpoints of unique feature values. They assign a single majority label to each leaf for internal FP/FN computations as described in the spec.
- The `ConditionNumberTree` computes provisional transfer matrices using a validation set; for small examples a simple heuristic mapping is used — see code comments for details.
- These implementations prioritize clarity and testability; for large-scale experiments you may want to replace them with optimized or C/C++ backed implementations.
