# models/random_forest.py
from datetime import datetime
import logging
import json
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

from ..utils.logger import init_logger

logger = init_logger("random_forest_logger")


class RandomForest:
    """
    A class for creating and managing a Random Forest model.
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        """
        Initialize the Random Forest model with the specified hyperparameters.

        Args:
            n_estimators (int): Number of trees in the forest. Defaults to 100.
            max_depth (int): Maximum depth of the tree. Defaults to None (no limit).
            random_state (int): Seed for the random number generator. Defaults to None.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        logger.debug(
            f"RandomForest initialized with n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state}"
        )

    def build_model(self):
        """
        Build the Random Forest model.
        """
        logger.info("Building Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            verbose=1,
        )
        logger.info("Random Forest model built.")

    def train(self, X_train, y_train):
        """
        Train the Random Forest model on the provided training data.

        Args:
            X_train (numpy.ndarray): Training input features.
            y_train (numpy.ndarray): Training target labels.
        """
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        logger.info("Random Forest model training completed.")

    def predict(self, X):
        """
        Make predictions using the trained Random Forest model.

        Args:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        logger.info("Making predictions with Random Forest model...")
        predictions = self.model.predict(X)
        logger.info("Predictions completed.")
        return predictions

    def save_model(self, output_dir):
        """
        Save the trained Random Forest model to a file.

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
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
        }
        with open(companion_path, "w") as file:
            json.dump(model_config, file, indent=4)

        logger.info(f"Random Forest model saved to {model_path}")
        logger.info(f"Model configuration saved to {companion_path}")
        return

    def load_model(self, model_path):
        """
        Load the trained Random Forest model from a file.

        Args:
            model_path (str): Path to the saved model file.
        """
        self.model = joblib.load(model_path)
        logger.info(f"Random Forest model loaded from {model_path}")
