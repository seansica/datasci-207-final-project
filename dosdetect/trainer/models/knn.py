# models/knn.py
from datetime import datetime
import logging
import json
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier

from ..utils.logger import init_logger

logger = init_logger("knn_logger")


class KNN:
    """
    A class for creating and managing a K-Nearest Neighbors (KNN) model.
    """

    def __init__(self, n_neighbors=5):
        """
        Initialize the KNN model with the specified number of neighbors.

        Args:
            n_neighbors (int): Number of neighbors to consider. Defaults to 5.
        """
        self.n_neighbors = n_neighbors
        self.model = None
        logger.debug(f"KNN initialized with n_neighbors={n_neighbors}")

    def build_model(self):
        """
        Build the KNN model.
        """
        logger.info("Building KNN model...")
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        logger.info("KNN model built.")

    def train(self, X_train, y_train):
        """
        Train the KNN model on the provided training data.

        Args:
            X_train (numpy.ndarray): Training input features.
            y_train (numpy.ndarray): Training target labels.
        """
        logger.info("Training KNN model...")
        self.model.fit(X_train, y_train)
        logger.info("KNN model training completed.")

    def predict(self, X):
        """
        Make predictions using the trained KNN model.

        Args:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        logger.info("Making predictions with KNN model...")
        predictions = self.model.predict(X)
        logger.info("Predictions completed.")
        return predictions

    def save_model(self, output_dir):
        """
        Save the trained KNN model to a file.

        Args:
            output_dir (str): Directory to save the model file.
        """
        # Ensure the target directory exists
        expanded_output_dir = os.path.expanduser(output_dir)
        try:
            os.makedirs(expanded_output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directories: {e}")

        model_path = os.path.join(expanded_output_dir, "model.pkl")
        joblib.dump(self.model, model_path)

        companion_path = os.path.join(expanded_output_dir, "hyperparameters.json")
        model_config = {
            "n_neighbors": self.n_neighbors
        }
        with open(companion_path, "w") as file:
            json.dump(model_config, file, indent=4)

        logger.info(f"KNN model saved to {model_path}")
        logger.info(f"Model configuration saved to {companion_path}")
        return

    def load_model(self, model_path):
        """
        Load the trained KNN model from a file.

        Args:
            model_path (str): Path to the saved model file.
        """
        self.model = joblib.load(model_path)
        logger.info(f"KNN model loaded from {model_path}")
