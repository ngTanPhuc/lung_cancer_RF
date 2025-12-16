from typing import List
import numpy as np
import pandas as pd

from .base_model import BaseModel
from .decision_tree import DecisionTreeModel


class AdaBoostModel(BaseModel):
    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models: List[DecisionTreeModel] = []
        self.alphas: List[float] = []

    # =======================================
    # TRAIN
    # =======================================
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        y_signed = y.map({0: -1, 1: 1}).values

        n_samples = len(y)
        sample_weights = np.ones(n_samples) / n_samples

        self.models = []
        self.alphas = []

        for _ in range(self.n_estimators):
            # ---- Weighted resampling ----
            indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weights
            )
            X_resampled = X.iloc[indices]
            y_resampled = y.iloc[indices]

            # ---- Train weak learner (decision stump) ----
            stump = DecisionTreeModel(max_depth=1)
            stump.train(X_resampled, y_resampled)

            # ---- Predict on original data ----
            y_pred = stump.predict(X)
            y_pred_signed = np.where(y_pred == 1, 1, -1)

            # ---- Weighted error ----
            err = np.sum(sample_weights * (y_pred_signed != y_signed))

            # Stop if learner is too weak
            if err <= 1e-10 or err >= 0.5:
                break

            # ---- Model weight ----
            alpha = self.learning_rate * 0.5 * np.log((1 - err) / err)

            # ---- Update sample weights ----
            sample_weights *= np.exp(-alpha * y_signed * y_pred_signed)
            sample_weights /= np.sum(sample_weights)

            self.models.append(stump)
            self.alphas.append(alpha)

        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_is_trained()

        X = X.reset_index(drop=True)
        agg = np.zeros(len(X))

        for model, alpha in zip(self.models, self.alphas):
            pred = model.predict(X)
            pred_signed = np.where(pred == 1, 1, -1)
            agg += alpha * pred_signed

        return np.where(agg >= 0, 1, 0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.check_is_trained()

        X = X.reset_index(drop=True)
        agg = np.zeros(len(X))

        for model, alpha in zip(self.models, self.alphas):
            pred = model.predict(X)
            pred_signed = np.where(pred == 1, 1, -1)
            agg += alpha * pred_signed

        proba_pos = 1 / (1 + np.exp(-2 * agg))
        proba_neg = 1 - proba_pos

        return np.vstack([proba_neg, proba_pos]).T