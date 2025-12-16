from .base_model import BaseModel
import numpy as np
import pandas as pd
from ..utils import LogisticLoss

class XGBoostTree:
    class TreeNode:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

        def is_leaf(self):
            return self.value is not None

    def __init__(self, reg_lambda = 1.0, gamma = 0.0, max_depth = 10,
                 max_leaf_nodes = 50, min_samples_split = 2):
        self.reg_lambda = reg_lambda  # complexity cost for num of leaves
        self.gamma = gamma  # L2 regularization for leaf's weight
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.root = None
        self.leaf_count = 0

    def _calculate_leaf_weight(self, g, h):
        """
        For every leaf j with dataset I_j
        G_j = sum(g_i), H_j = sum(h_i) for i in I_j
        :param g: list of g_i in I_j
        :param h: list of h_i in I_j
        :return: w*_j = - G_j / (H_j + reg_lambda)
        """
        G_j = np.sum(g)
        H_j = np.sum(h)
        return - G_j / (H_j + self.reg_lambda)

    def _calculate_Gain(self, G_L, H_L, G_R, H_R, G_root, H_root):
        """
        Calculate Gain of the tree:
        0.5 * [G_L^2/(H_L+reg_lambda) + G_R^2/(H_R+reg_lambda) - (G_L+G_R)^2/(H_L+H_R+reg_lambda)] - gamma
        """
        # TODO
        score_left = G_L**2 / (H_L + self.reg_lambda)
        score_right = G_R**2 / (H_R + self.reg_lambda)
        score_both = (G_L + G_R)**2 / (H_L + H_R + self.reg_lambda)
        return 0.5 * (score_left + score_right - score_both) - self.gamma

    def _best_split(self, X, g, h):
        """
        Find the best split for a leaf node
        :return: best_idx, best_threshold
        """
        m, n = X.shape  # m: number of samples, n: number of features

        best_gain = -float('inf')
        best_idx = None
        best_threshold = None

        # Loop through every feature and calculate the gain of that feature
        for feature_idx in range(n):
            # Get the unique values of that feature. If it's boolean (int 0 and 1) => thresholds = [0, 1].
            # If it's int (AGE) => thresholds = [ages...]
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                # Return a boolean array
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                # Skip this threshold if one side is empty (all values are False)
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                G_L = g[left_mask].sum()
                H_L = h[left_mask].sum()
                G_R = g[right_mask].sum()
                H_R = h[right_mask].sum()

                gain = self._calculate_Gain(G_L, H_L, G_R, H_R, G_R, H_R)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = feature_idx
                    best_threshold = threshold

        return best_idx, best_threshold

    def _build_tree(self, X, g, h, depth):
        # Stop recursion condition
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            leaf_weight = self._calculate_leaf_weight(g, h)
            return self.TreeNode(value=leaf_weight)

        # Find best split
        best_idx, best_threshold = self._best_split(X, g, h)
        if best_idx is None:
            leaf_weight = self._calculate_leaf_weight(g, h)
            return self.TreeNode(value=leaf_weight)

        # Split data
        left_mask = X[:, best_idx] <= best_threshold
        right_mask = ~left_mask
        X_left = X[left_mask]
        X_right = X[right_mask]
        g_left = g[left_mask]
        g_right = g[right_mask]
        h_left = h[left_mask]
        h_right = h[right_mask]

        left_node = self._build_tree(X_left, g_left, h_left, depth + 1)
        right_node = self._build_tree(X_right, g_right, h_right, depth + 1)
        return self.TreeNode(feature=best_idx, threshold=best_threshold, left=left_node, right=right_node)

    def fit(self, X, g ,h):
        self.root = self._build_tree(X, g, h, 0)

    def _predict_one(self, x, node: TreeNode):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(row, self.root) for row in X.values])

class XGBoost(BaseModel):
    """
    Parameters:
        n_estimators (M): the number of trees in the forest
        lr (eta): learning rate
        min_samples_split: minimum number of samples required to split this tree
        min_impurity: the minimum impurity required to split this tree
        max_depth: maximum depth of the tree
    """
    def __init__(self, reg_lambda: float = 1.0, gamma: float = 0.0, max_leaf_nodes: int = 50,
                 n_estimators: int = 100, lr: float = 0.2, min_samples_split: int = 5,
                 min_impurity: float = 0.01, max_depth: int = 10):
        super().__init__()
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.max_leaf_nodes = max_leaf_nodes
        self.n_estimators = n_estimators
        self.lr = lr
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity = min_impurity

        self.loss = LogisticLoss()

        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(XGBoostTree(
                self.reg_lambda,
                self.gamma,
                self.max_depth,
                self.max_leaf_nodes,
                self.min_samples_split,
            ))

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_data = X.values if isinstance(X, pd.DataFrame) else X
        y_data = y.values if isinstance(y, pd.Series) else y

        curr_pred = np.zeros(X_data.shape[0])

        for tree in self.trees:
            p = self.loss.calc_p(curr_pred)
            g = self.loss.gradient(y_data, curr_pred)
            h = self.loss.hessian(curr_pred)

            tree.fit(X_data, g, h)

        self.is_trained = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.check_is_trained()

        pred = np.zeros(X.shape[0])
        for tree in self.trees:
            pred += self.lr * tree.predict(X)

        return self.loss.calc_p(pred)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        propas = self.predict_proba(X)
        return (propas > 0.5).astype(int)





