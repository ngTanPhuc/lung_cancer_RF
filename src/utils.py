import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models import BaseModel

class ModelIO:
    """
    Utility class for saving, loading, and listing machine learning models.
    Models are serialized using pickle and stored inside the project-level `checkpoints/` directory.
    """

    @staticmethod
    def _get_project_root():
        """
        Return the absolute path to the project root directory.
        Assumes this file is located inside `utils/`.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, ".."))

    @staticmethod
    def save(model: BaseModel, filename: str = "model.pkl") -> bool:
        """
        Serialize and save a model to the `checkpoints/` directory.

        Parameters
        ----------
        model : BaseModel
            The trained model to be saved.
        filename : str
            Name of the output file (must end with .pkl).
        """
        project_root = ModelIO._get_project_root()
        save_dir = os.path.join(project_root, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, filename)

        with open(save_path, "wb") as f:
            pickle.dump(model, f)

        project_name = os.path.basename(project_root)
        relative_path = os.path.join(project_name, "checkpoints", filename)
        print(f"Model saved to: {relative_path}")
        return True

    @staticmethod
    def load(filename: str = "model.pkl") -> BaseModel:
        """
        Load a serialized model from the `checkpoints/` directory.

        Parameters
        ----------
        filename : str
            Name of the model file to load.

        Returns
        -------
        BaseModel
            The deserialized model object.
        """
        project_root = ModelIO._get_project_root()
        load_path = os.path.join(project_root, "checkpoints", filename)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model not found at: {load_path}")

        with open(load_path, "rb") as f:
            model = pickle.load(f)

        project_name = os.path.basename(project_root)
        relative_path = os.path.join(project_name, "checkpoints", filename)
        print(f"Model loaded from: {relative_path}")
        return model

    @staticmethod
    def delete(filename: str = "model.pkl") -> bool:
        """
        Delete a model file from the `checkpoints/` directory.

        Parameters
        ----------
        filename : str
            Name of the model file to delete.

        Raises
        ------
        FileNotFoundError
            If the specified model file does not exist.
        """
        project_root = ModelIO._get_project_root()
        model_path = os.path.join(project_root, "checkpoints", filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        os.remove(model_path)
        print(f"Model deleted: {model_path}")
        return True

    @staticmethod
    def exists(filename: str = "model.pkl") -> bool:
        """
        Check whether a given model file exists in the `checkpoints/` directory.

        Parameters
        ----------
        filename : str
            Name of the model file to check.

        Returns
        -------
        bool
            True if the file exists, otherwise False.

        Use case:
        - Before attempting to load or delete a model.
        - Useful in UI applications, APIs, or training pipelines.
        """
        project_root = ModelIO._get_project_root()
        model_path = os.path.join(project_root, "checkpoints", filename)

        return os.path.exists(model_path)

    @staticmethod
    def list_models():
        """
        Return a list of saved model filenames in the `checkpoints/` directory.

        Returns
        -------
        list[str]
            List of `.pkl` model filenames.
            Returns an empty list if the directory does not exist or contains no checkpoints.
        """
        project_root = ModelIO._get_project_root()
        models_dir = os.path.join(project_root, "checkpoints")

        if not os.path.exists(models_dir):
            return []

        return [
            file
            for file in os.listdir(models_dir)
            if file.endswith(".pkl")
        ]

class LogisticLoss:
    def __init__(self):
        pass

    def calc_p(self, y_pred):
        return 1 / (1 + np.exp(-y_pred))

    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-5, 1 - 1e-5)  # to make sure the value is in between (1e-5 and 0.99999)
        p = self.calc_p(y_pred)
        return -(y_true*np.log(p) + (1 - y_true)*np.log(1 - p))

    def gradient(self, y_true, y_pred):  # g_i
        p = self.calc_p(y_pred)
        return p - y_true

    def hessian(self, y_pred):  # h_i
        p = self.calc_p(y_pred)
        return p*(1 - p)
    

class ModelComparator:
    def __init__(self):
        self.models: list[BaseModel] = []

    def load_model(self, filename: str):
        """
        Load a saved BaseModel using ModelIO and add it to comparator.
        """
        model = ModelIO.load(filename)

        if not isinstance(model, BaseModel):
            raise TypeError("Loaded object is not a BaseModel.")

        '''
        required_attrs = ["acc", "prec", "rec", "f1"]
        for attr in required_attrs:
            if not hasattr(model, attr):
                raise RuntimeError(
                    f"Model '{model.model_name}' has no '{attr}'. Call score() before saving."
                )
        '''

        self.models.append(model)

    def plot_comparison(self, metrics=None):
        if not self.models:
            raise RuntimeError("No models loaded.")

        if metrics is None:
            metrics = ["acc", "prec", "rec", "f1", "auc"]

        # ---------- fixed color per model ----------
        cmap = plt.get_cmap("tab10")
        model_colors = {
            m.model_name: cmap(i)
            for i, m in enumerate(self.models)
        }

        plt.figure(figsize=(12, 6))

        xticks = []
        xticklabels = []
        x_cursor = 0.0

        for metric in metrics:
            values = []
            names = []

            for m in self.models:
                v = getattr(m, metric, None)
                if v is not None:
                    values.append(v)
                    names.append(m.model_name)

            if not values:
                continue

            n = len(values)
            width = 0.8 / n
            offsets = [x_cursor + i * width for i in range(n)]

            for off, val, name in zip(offsets, values, names):
                bar = plt.bar(
                    off,
                    val,
                    width,
                    color=model_colors[name]
                )
                plt.text(
                    off,
                    val + 0.015,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10
                )

            center = x_cursor + width * (n - 1) / 2
            xticks.append(center)
            xticklabels.append(metric.upper())

            x_cursor += 1.0  # spacing between metrics

        # ---------- axes ----------
        plt.xticks(xticks, xticklabels)
        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        plt.title("Model Performance Comparison")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # ---------- legend ----------
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=model_colors[m.model_name])
            for m in self.models
        ]
        plt.legend(handles, [m.model_name for m in self.models], loc="lower left")

        plt.tight_layout()
        plt.show()
