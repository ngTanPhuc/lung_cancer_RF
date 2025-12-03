from .base_model import BaseModel
from .decision_tree import DecisionTreeModel

import numpy as np
import os
import pickle

class RandomForestModel(BaseModel):
    def __init__(self, n_trees=10, random_state=6):
        self.n_trees = n_trees
        self.random_state = random_state
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

    def train(self, X, y):
        self.trees = []
        self.oob_predictions = {i: [] for i in range(len(X))}

        for i in range(self.n_trees):
            tree = DecisionTreeModel(is_forest_tree=True)

            X_sample, y_sample, inbag_idx, oob_idx = self._bootstrap_sample(X, y)
            tree.train(X_sample, y_sample)
            self.trees.append(tree)

            if len(oob_idx) > 0:
                preds = tree.predict(X.iloc[oob_idx])
                for idx, p in zip(oob_idx, preds):
                    self.oob_predictions[idx].append(p)

            print(f"✓ Trained tree {i + 1}/{self.n_trees}")

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

    def predict(self, X):
        all_preds = []
        for tree in self.trees:
            all_preds.append(tree.predict(X))
        return self._majority_vote(all_preds)
    
    def save(self, filename="random_forest_model.pkl"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, filename)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {save_path}")

    @staticmethod
    def load(filename="random_forest_model.pkl"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        load_path = os.path.join(project_root, "models", filename)
        with open(load_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from: {load_path}")
        return model