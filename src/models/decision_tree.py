import numpy as np

from .base_model import BaseModel

class DecisionTreeModel(BaseModel):
    class TreeNode:
        def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
            self.feature = feature
            self.gini = None
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

        def is_leaf(self):
            return self.value is not None

    def __init__(self, is_forest_tree = False, max_depth = None, max_leaf_nodes = None, min_samples_leaf = None):
        super.__init__()
        self.is_forest_tree = is_forest_tree

        self.max_depth = max_depth if max_depth else int("inf")
        self.max_leaf_nodes = max_leaf_nodes if max_leaf_nodes else int("inf")
        self.min_samples_leaf = min_samples_leaf if min_samples_leaf else int("inf")
        self.leaf_count = 0

        self.root = None

    # =======================================
    # OVERIDE TRAIN
    # =======================================
    def _select_feature_subset(self, n_features):
        if self.is_forest_tree:
            k = max(int(np.sqrt(n_features)), 1)  
            return np.random.choice(n_features, k, replace=False) 
        else:
            return np.arange(n_features)
            
    def _gini(self, y):
        label_type, type_count = np.unique(y, return_counts=True)
        p = type_count / len(y)
        return 1 - np.sum(p**2)

    def _best_split(self, X, y, features):
        best_feature = None
        best_threshold = None
        best_impurity = float("inf")

        n_samples = len(y)

        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                child_impurity = (left_mask.sum()*left_gini + right_mask.sum()*right_gini) / n_samples

                if child_impurity < best_impurity:
                    best_impurity = child_impurity
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build(self, X, y, depth):
        if not self.is_forest_tree:
            if len(np.unique(y)) == 1:
                self.leaf_count += 1
                return self.TreeNode(value=np.argmax(np.bincount(y)))
            # max_depth
            if depth >= self.max_depth:
                self.leaf_count += 1
                return self.TreeNode(value=np.argmax(np.bincount(y)))
            # min_samples_split
            if len(y) < self.min_samples_leaf:
                self.leaf_count += 1
                return self.TreeNode(value=np.argmax(np.bincount(y)))
            # max_leaf_nodes
            if self.leaf_count >= self.max_leaf_nodes:
                self.leaf_count += 1
                return self.TreeNode(value=np.argmax(np.bincount(y)))
        else:
            if len(np.unique(y)) == 1:
                self.leaf_count += 1
                return self.TreeNode(value=np.argmax(np.bincount(y)))

        n_features = X.shape[1]
        features = self._select_feature_subset(n_features)
        feature, threshold = self._best_split(X, y, features)

        if feature is None:
            self.leaf_count += 1
            return self.TreeNode(value=np.argmax(np.bincount(y)))

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left_node = self._build(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build(X[right_mask], y[right_mask], depth + 1)
        return self.TreeNode(feature=feature, threshold=threshold, left=left_node, right=right_node)

    def train(self, X, y):
        self.root = self._build(X.values, y.values, 0)
        self.is_trained = True
        return

    # =======================================
    # OVERIDE PREDICT - PREDICT_PROBA
    # =======================================
    def _predict_one(self, x, node: TreeNode):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        self.check_is_trained()
        return np.array([self._predict_one(row, self.root) for row in X.values])
    
    def predict_proba(self, X):
        self.check_is_trained()
        return super().predict_proba(X)