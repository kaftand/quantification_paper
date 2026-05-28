# new_experiment/impl

Reference implementation for the Decision Tree based quantification experiments.

Prerequisites
- Activate the `quant` conda environment as requested:

```powershell
conda activate quant
```

Quick usage

```python
from new_experiment.impl import (
    ClassificationTree,
    HoldoutTransferMatrixEstimator,
    EMQuantificationSolver,
    TreeBinQuantifier,
)

# build objects
tree = ClassificationTree(max_depth=3)
est = HoldoutTransferMatrixEstimator()
solver = EMQuantificationSolver()
q = TreeBinQuantifier(tree, est, solver)

# Fit and quantify (example arrays X_train, y_train, X_val, y_val, X_test)
# q.fit(X_train, y_train, X_val, y_val)
# pi = q.quantify(X_test)
```

This is a minimal, readable reference: for production experiments adapt choices (solvers,
regularization, missing-class handling) to your needs.
