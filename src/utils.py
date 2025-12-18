import os
import pickle
import numpy as np

from .models import BaseModel

class ModelIO:
    """
    Utility class for saving, loading, and listing machine learning models.
    Models are serialized using pickle and stored inside the project-level `models/` directory.
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
        Serialize and save a model to the `models/` directory.

        Parameters
        ----------
        model : BaseModel
            The trained model to be saved.
        filename : str
            Name of the output file (must end with .pkl).
        """
        project_root = ModelIO._get_project_root()
        save_dir = os.path.join(project_root, "models")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, filename)

        with open(save_path, "wb") as f:
            pickle.dump(model, f)

        print(f"Model saved to: {save_path}")
        return True

    @staticmethod
    def load(filename: str = "model.pkl") -> BaseModel:
        """
        Load a serialized model from the `models/` directory.

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
        load_path = os.path.join(project_root, "models", filename)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model not found at: {load_path}")

        with open(load_path, "rb") as f:
            model = pickle.load(f)

        print(f"Model loaded from: {load_path}")
        return model

    @staticmethod
    def delete(filename: str = "model.pkl") -> bool:
        """
        Delete a model file from the `models/` directory.

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
        model_path = os.path.join(project_root, "models", filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        os.remove(model_path)
        print(f"Model deleted: {model_path}")
        return True

    @staticmethod
    def exists(filename: str = "model.pkl") -> bool:
        """
        Check whether a given model file exists in the `models/` directory.

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
        model_path = os.path.join(project_root, "models", filename)

        return os.path.exists(model_path)

    @staticmethod
    def list_models():
        """
        List all saved model files in the `models/` directory.

        Notes
        -----
        - Only `.pkl` files are considered valid models.
        - This function is useful for inspecting trained models available for loading.
        """
        project_root = ModelIO._get_project_root()
        models_dir = os.path.join(project_root, "models")

        if not os.path.exists(models_dir):
            print("Folder \"models/\" not found.")
            return

        files = os.listdir(models_dir)
        model_files = [file for file in files if file.endswith(".pkl")]

        print("List of available models:")
        for m in model_files:
            print(f" - {m}")
        return model_files

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