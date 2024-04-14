# models/logistic_regression.py
import json
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from ..utils.logger import init_logger

logger = init_logger(__name__)


class LogisticRegressionModel:
    """
    A class for creating and managing a Logistic Regression model.
    """

    def __init__(self, C=1.0, max_iter=100, random_state=None, auto_tune=False):
        """
        Initialize the Logistic Regression model with the specified hyperparameters.

        Args:
            C (float): Inverse of regularization strength. Defaults to 1.0.
            max_iter (int): Maximum number of iterations for solver. Defaults to 100.
            random_state (int): Seed for the random number generator. Defaults to None.
            auto_tune (bool): Whether to use GridSearchCV for hyperparameter tuning. Defaults to False.
        """
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.auto_tune = auto_tune
        self.model = None
        logger.debug(
            f"LogisticRegressionModel initialized with C={C}, max_iter={max_iter}, random_state={random_state}, auto_tune={auto_tune}"
        )

    def build_model(self):
        """
        Build the Logistic Regression model.
        """
        logger.info("Building Logistic Regression model...")
        self.model = LogisticRegression()
        logger.info("Logistic Regression model built.")

    def train(self, X_train, y_train):
        """
        Train the Logistic Regression model on the provided training data.

        Args:
            X_train (numpy.ndarray): Training input features.
            y_train (numpy.ndarray): Training target labels.
        """
        logger.info("Training Logistic Regression model...")

        if self.auto_tune:
            # Define the hyperparameter grid for tuning
            param_grid = {
                "C": [0.1, 1.0, 10.0],
                "max_iter": [100, 500, 1000],
            }

            # Perform grid search with cross-validation
            grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="accuracy")
            grid_search.fit(X_train, y_train)

            # Set the best hyperparameters found by grid search
            self.model = grid_search.best_estimator_
            self._best_params = grid_search.best_params_
            logger.info(f"Best hyperparameters: {grid_search.best_params_}")
        else:
            self.model.set_params(
                C=self.C, max_iter=self.max_iter, random_state=self.random_state
            )
            self.model.fit(X_train, y_train)

        logger.info("Logistic Regression model training completed.")

    def predict(self, X):
        """
        Make predictions using the trained Logistic Regression model.

        Args:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        logger.info("Making predictions with Logistic Regression model...")
        predictions = self.model.predict(X)
        logger.info("Predictions completed.")
        return predictions

    def save_model(self, output_dir):
        """
        Save the trained Logistic Regression model to a file.

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
        model_config = (
            self._best_params
            if self.auto_tune
            else {
                "C": self.C,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
            }
        )
        with open(companion_path, "w") as file:
            json.dump(model_config, file, indent=4)

        logger.info(f"Logistic Regression model saved to {model_path}")

    def load_model(self, model_path):
        """
        Load the trained Logistic Regression model from a file.

        Args:
            model_path (str): Path to the saved model file.
        """
        self.model = joblib.load(model_path)
        logger.info(f"Logistic Regression model loaded from {model_path}")
