from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

class BaseModel(ABC):
    """
    Minimal abstract base class for ML models.
    Defines the required interface for training, prediction,
    evaluation, and parameter management.
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        # Name of the concrete model class (useful for logging and errors)
        self.model_name: str = self.__class__.__name__

        # Optional integer seed for reproducibility
        self.random_state: Optional[int] = random_state

        # Indicates whether the model has been trained
        self.is_trained: bool = False

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model on the given dataset.

        Use case:
        - Implemented by each concrete model (e.g., DecisionTree, RandomForest).
        - Responsible for learning patterns from X and y.
        """
        pass

    def check_is_trained(self):
        """
        Verify that the model has been trained before performing inference.

        Use case:
        - Should be called inside predict() and predict_proba() to prevent accidental inference on an untrained model.
        - Helps ensure correctness and avoid undefined behavior.
        """
        if not self.is_trained:
            raise RuntimeError(
                f"{self.model_name} is not trained yet. Call train() first."
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Convenience method that calls train() for API consistency.

        Use case:
        - Allows sklearn-style usage: model.fit(X, y)
        - Alias of train().
        - Does NOT automatically update is_trained; the model subclass should set is_trained = True after training is complete.
        """
        self.train(X, y)
        return

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return predicted class labels for input X.

        Use case:
        - Used for inference after training.
        - Must return a numpy array of shape (n_samples,).
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return class probability estimates for input X.

        Use case:
        - Provides probabilities instead of hard labels.
        - Needed for ROC/AUC, threshold tuning, or risk scoring.
        - Should be implemented only if the model naturally supports probability output.
        """
        raise NotImplementedError("This model does not support probability prediction.")

    def score(self, X, y, positive_label: int = 1) -> None:
        """
        Print classification evaluation report and
        store metrics as model attributes.
        """

        # ---------- y_true ----------
        y_true = np.array(y).astype(int).ravel()

        # ---------- predict ----------
        y_pred_raw = np.array(self.predict(X))

        if y_pred_raw.ndim == 2 and y_pred_raw.shape[1] >= 2:
            y_pred = np.argmax(y_pred_raw, axis=1)

        elif y_pred_raw.ndim == 1 and np.issubdtype(y_pred_raw.dtype, np.floating):
            uniq = set(np.unique(y_pred_raw).tolist())
            if not uniq.issubset({0.0, 1.0}):
                y_pred = (y_pred_raw >= 0.5).astype(int)
            else:
                y_pred = y_pred_raw.astype(int)

        else:
            y_pred = y_pred_raw.astype(int).ravel()

        # ---------- Confusion Matrix ----------
        labels = [0, 1]
        cm = np.zeros((2, 2), dtype=int)

        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        self.confusion_matrix = cm

        cm_df = pd.DataFrame(
            cm,
            index=pd.Index(labels, name="Actual"),
            columns=pd.Index(labels, name="Predicted")
        )

        print("\nConfusion Matrix:")
        print(cm_df.to_string())

        TN, FP, FN, TP = cm.ravel()
        print(f"\nTN={TN}, FP={FP}, FN={FN}, TP={TP}")

        # ---------- Metrics ----------
        self.acc = (TP + TN) / len(y_true) if len(y_true) > 0 else 0.0
        self.rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        self.prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        self.f1 = (
            2 * self.prec * self.rec / (self.prec + self.rec)
            if (self.prec + self.rec) > 0 else 0.0
        )

        print(f"\nAccuracy : {self.acc:.4f}")
        print(f"Recall   : {self.rec:.4f}   (TP/(TP+FN))")
        print(f"Precision: {self.prec:.4f}   (TP/(TP+FP))")
        print(f"F1-score : {self.f1:.4f}   (2*P*R/(P+R))")
        
        # ---------- AUC ----------
        self.auc = None
        try:
            y_score = self.predict_proba(X)

            if y_score.ndim == 2:
                y_score = y_score[:, positive_label]

            idx = np.argsort(-y_score)
            y_true_sorted = y_true[idx]

            P = np.sum(y_true_sorted == positive_label)
            N = len(y_true_sorted) - P

            tp = fp = 0
            tpr = []
            fpr = []

            for label in y_true_sorted:
                if label == positive_label:
                    tp += 1
                else:
                    fp += 1

                tpr.append(tp / P if P > 0 else 0)
                fpr.append(fp / N if N > 0 else 0)

            auc = 0.0
            for i in range(1, len(tpr)):
                auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2

            self.auc = auc
            print(f"AUC      : {self.auc:.4f}")

        except (NotImplementedError, AttributeError):
            print("AUC      : N/A (predict_proba not supported)")

    def get_params(self):
        """
        Return a dictionary of all public model attributes.

        Use case:
        - Useful for logging, debugging, checkpointing, and hyperparameter tuning.
        - Excludes private attributes (those starting with '_').
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def set_params(self, **params):
        """
        Dynamically set model attributes via keyword arguments.

        Use case:
        - Enables automated hyperparameter tuning and grid search.
        - Allows external configuration without manually editing attributes.
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self
    
    def plot_evaluation(self) -> None:
        """
        Plot Confusion Matrix and Evaluation Metrics.
        Requires score() to be called beforehand.
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import colors

        # ---------- Check ----------
        if not hasattr(self, "confusion_matrix"):
            raise RuntimeError("Confusion matrix not found. Call score() first.")

        cm = self.confusion_matrix

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # ==================================================
        # Confusion Matrix
        # ==================================================
        ax = axes[0]

        norm = colors.Normalize(vmin=0, vmax=cm.max())
        im = ax.imshow(cm, cmap="YlGnBu", norm=norm)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
                ax.text(
                    j, i, cm[i, j],
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color=color
                )

        # ==================================================
        # Metrics Bar Chart
        # ==================================================
        ax = axes[1]

        metrics = {
            "Accuracy": self.acc,
            "Precision": self.prec,
            "Recall": self.rec,
            "F1": self.f1,
        }

        if hasattr(self, "auc") and self.auc is not None:
            metrics["AUC"] = self.auc

        names = list(metrics.keys())
        values = list(metrics.values())

        bars = ax.bar(names, values)
        ax.set_ylim(0, 1.05)
        ax.set_title("Evaluation Metrics")

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold"
            )

        plt.tight_layout()
        plt.show()
