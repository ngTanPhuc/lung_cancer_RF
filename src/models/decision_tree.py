from typing import Optional, override
import numpy as np
from .base_model import BaseModel


class DecisionTreeModel(BaseModel):
    class TreeNode:
        def __init__(
            self,
            *,
            feature: Optional[int] = None,
            feature_name: Optional[str] = None,
            gini: Optional[float] = None,
            threshold: Optional[float] = None,
            left: Optional["DecisionTreeModel.TreeNode"] = None,
            right: Optional["DecisionTreeModel.TreeNode"] = None,
            value: Optional[int] = None,
        ):
            self.feature = feature
            self.feature_name = feature_name
            self.gini = gini
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

        def is_leaf(self) -> bool:
            return self.value is not None

    def __init__(
        self,
        is_forest_tree: bool = False,
        max_depth: int | None = None,
        max_leaf_nodes: int | None = None,
        min_samples_leaf: int | None = None,
    ):
        super().__init__()
        self.root = None
        self.is_forest_tree = is_forest_tree

        self.max_depth = max_depth
        self.depth = 0

        self.max_leaf_nodes = max_leaf_nodes
        self.leaf_count = 0

        self.min_samples_leaf = min_samples_leaf

        self.feature_names = None

    # =======================================
    # TRAIN
    # =======================================
    def _select_feature(self, n_features: int) -> np.ndarray:
        if self.is_forest_tree:
            k = max(int(np.sqrt(n_features)), 1)
            return np.random.choice(np.arange(n_features), k, replace=False)
        return np.arange(n_features)

    def _gini(self, y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray, features: np.ndarray):
        best_feature = None
        best_threshold = None
        best_impurity = float("inf")

        n_samples = len(y)

        for feature in features:
            feature = int(feature) 
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                if self.min_samples_leaf is not None:
                    if (
                        left_mask.sum() < self.min_samples_leaf
                        or right_mask.sum() < self.min_samples_leaf
                    ):
                        continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])

                impurity = (
                    left_mask.sum() * left_gini
                    + right_mask.sum() * right_gini
                ) / n_samples

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_impurity

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        assert isinstance(X, np.ndarray), "_build expects numpy array"

        self.depth = max(self.depth, depth)

        # ---- stop conditions ----
        if len(np.unique(y)) == 1:
            self.leaf_count += 1
            return self.TreeNode(value=int(np.argmax(np.bincount(y))))

        if self.max_depth is not None and depth >= self.max_depth:
            self.leaf_count += 1
            return self.TreeNode(value=int(np.argmax(np.bincount(y))))

        if self.max_leaf_nodes is not None:
            if self.leaf_count + 1 >= self.max_leaf_nodes:
                self.leaf_count += 1
                return self.TreeNode(value=int(np.argmax(np.bincount(y))))

        # ---- split ----
        n_features = X.shape[1]
        features = self._select_feature(n_features)
        feature, threshold, impurity = self._best_split(X, y, features)

        if feature is None:
            self.leaf_count += 1
            return self.TreeNode(value=int(np.argmax(np.bincount(y))))

        feature = int(feature)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_node = self._build(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build(X[right_mask], y[right_mask], depth + 1)

        return self.TreeNode(
            feature=feature,
            feature_name=self.feature_names[feature],
            threshold=threshold,
            gini=impurity,
            left=left_node,
            right=right_node,
        )

    @override
    def train(self, X, y):
        self.depth = 0
        self.leaf_count = 0
        self.feature_names = list(X.columns)

        self.root = self._build(X.values, y.values)
        self.is_trained = True

    # =======================================
    # PREDICT
    # =======================================
    def _predict_one(self, x: np.ndarray, node: TreeNode):
        if node.is_leaf():
            return node.value

        assert isinstance(node.feature, (int, np.integer)), (
            f"BUG: feature={node.feature}, type={type(node.feature)}"
        )

        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    @override
    def predict(self, X):
        self.check_is_trained()
        return np.array([self._predict_one(row, self.root) for row in X.values])

    @override
    def predict_proba(self, X):
        self.check_is_trained()
        return super().predict_proba(X)

    # =======================================
    # SHOW TREE
    # =======================================
    def print_tree(self) -> None:
        """
        Print the full decision tree structure (no depth limit).
        """
        self.check_is_trained()
        self._print_node(self.root, prefix="", is_last=True)


    def _print_node(self, node: TreeNode, prefix: str, is_last: bool) -> None:
        connector = "└── " if is_last else "├── "

        if node.is_leaf():
            print(f"{prefix}{connector}Predict: {node.value}")
            return

        feature_name = (
            node.feature_name
            if node.feature_name is not None
            else f"feature_{node.feature}"
        )

        print(
            f"{prefix}{connector}"
            f"[{feature_name} <= {node.threshold:.4f}] "
            f"(gini={node.gini:.4f})"
        )

        # prefix for children
        child_prefix = prefix + ("    " if is_last else "│   ")

        # left = not last (because right exists)
        self._print_node(node.left, child_prefix, is_last=False)

        # right = last child
        self._print_node(node.right, child_prefix, is_last=True)
