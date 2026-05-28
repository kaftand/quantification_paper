from sklearn.tree import DecisionTreeClassifier
import numpy as np


class ClassificationTree:
    """Thin wrapper around sklearn DecisionTreeClassifier providing
    the minimal API required by the experiment specs.
    """

    def __init__(self, max_depth=None, min_samples_leaf=1, criterion='gini', random_state=None):
        self.max_depth = max_depth
        self.min_leaf_size = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.tree_ = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_size,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        self.tree_.fit(X, y)
        self.classes_ = getattr(self.tree_, "classes_")
        self.n_classes_ = len(self.classes_)
        try:
            self.n_leaves_ = self.tree_.get_n_leaves()
        except Exception:
            # fallback to counting unique leaf indices on training data
            self.n_leaves_ = len(np.unique(self.tree_.apply(X)))
        return self

    def get_leaf_indices(self, X):
        """Return the tree node id (leaf id) for each sample in X."""
        return self.tree_.apply(X)
