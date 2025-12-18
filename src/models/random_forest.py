
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, override

from .base_model import BaseModel
from .decision_tree import DecisionTreeModel

class RandomForestModel(BaseModel):
    def __init__(self, n_trees = 10, random_state = 42):
        super().__init__(random_state = random_state)
        self.n_trees = n_trees
        self.rng = np.random.default_rng(random_state)
        self.trees = []
        self.oob_predictions = {}

    def _bootstrap_sample(self, X, y):
        n = len(X)
        inbag_idx = self.rng.integers(0, n, n)

        oob_mask = np.ones(n, dtype=bool)
        oob_mask[inbag_idx] = False
        oob_idx = np.where(oob_mask)[0]

        return X.iloc[inbag_idx], y.iloc[inbag_idx], inbag_idx, oob_idx

    def _majority_vote(self, predictions):
        preds = np.array(predictions)
        final = []
        for col in preds.T:
            values, counts = np.unique(col, return_counts=True)
            final.append(values[np.argmax(counts)])
        return np.array(final)

    @override
    def train(self, X, y):
        self.trees = []
        for i in range(len(X)):
            self.oob_predictions[i] = [] 

        for i in range(self.n_trees):
            tree = DecisionTreeModel(is_forest_tree=True)
            X_sample, y_sample, inbag_idx, oob_idx = self._bootstrap_sample(X, y)
            tree.train(X_sample, y_sample)
            self.trees.append(tree)

            if len(oob_idx) > 0:
                preds = tree.predict(X.iloc[oob_idx])
                for idx, pred in zip(oob_idx, preds):
                    self.oob_predictions[idx].append(pred)

        self.oob = self.oob_score(y)
        self.is_trained = True

    def oob_score(self, y):
        final_preds = []
        true_labels = []
        for idx, preds in self.oob_predictions.items():
            if len(preds) == 0:
                continue  
            values, counts = np.unique(preds, return_counts=True)
            majority = values[np.argmax(counts)]
            final_preds.append(majority)
            true_labels.append(y.iloc[idx])
        final_preds = np.array(final_preds)
        true_labels = np.array(true_labels)
        return np.mean(final_preds == true_labels)

    @override
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    @override
    def predict_proba(self, X):
        self.check_is_trained()
        n_samples = len(X)
        n_classes = 2  

        proba_sum = np.zeros((n_samples, n_classes))

        for tree in self.trees:
            tree_preds = tree.predict(X)
            for i, pred in enumerate(tree_preds):
                proba_sum[i, pred] += 1

        proba = proba_sum / len(self.trees)
        return proba