from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

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

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return class probability estimates for input X.

        Use case:
        - Provides probabilities instead of hard labels.
        - Needed for ROC/AUC, threshold tuning, or risk scoring.
        - Should be implemented only if the model naturally supports probability output.
        """
        raise NotImplementedError("This model does not support probability prediction.")

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Compute accuracy on a given dataset.

        Use case:
        - Helps detect overfitting and tune hyperparameters.
        - Typically applied to the test set for unbiased performance measurement.
        """
        return np.mean(self.predict(X) == y)

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