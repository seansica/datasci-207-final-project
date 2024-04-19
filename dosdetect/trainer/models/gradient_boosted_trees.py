from datetime import datetime
import os
import json
import logging
import xgboost as xgb
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from ..utils.logger import init_logger

logger = init_logger(__name__)


class GradientBoostedTreesModel:
    """
    A class for creating and managing a Gradient Boosted Trees model using XGBoost and Keras.
    """

    def __init__(self, input_shape, num_classes, auto_tune=True):
        """
        Initialize the Gradient Boosted Trees model with input shape and number of classes.

        Args:
            input_shape (tuple): Shape of the input data.
            num_classes (int): Number of output classes.
            auto_tune (bool): Whether to use hyperparameter autotuning. Defaults to True.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.auto_tune = auto_tune
        self.model = None
        logger.debug(
            f"Gradient Boosted Trees model initialized with input shape: {input_shape}, num_classes: {num_classes}, auto_tune: {auto_tune}"
        )

    def build_model(self, max_depth=3, learning_rate=0.1, n_estimators=100):
        """
        Build the Gradient Boosted Trees model.

        Args:
            max_depth (int): Maximum depth of the individual trees.
            learning_rate (float): Learning rate for boosting.
            n_estimators (int): Number of trees to fit.
        """
        logger.info("Building Gradient Boosted Trees model...")

        # Define the input layer
        input_layer = Input(shape=self.input_shape)

        # Create a dummy dense layer to match the input shape
        dense = Dense(1, activation='linear')(input_layer)

        # Create the XGBoost model
        self.xgb_model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective="multi:softprob",
            num_class=self.num_classes,
            tree_method="auto",
        )

        # Wrap the XGBoost model in a Keras model
        self.model = Model(inputs=input_layer, outputs=dense)

        logger.debug("Gradient Boosted Trees model architecture:")
        logger.debug(f"Max Depth: {max_depth}")
        logger.debug(f"Learning Rate: {learning_rate}")
        logger.debug(f"Number of Estimators: {n_estimators}")

        logger.info("Gradient Boosted Trees model built.")

    def compile_model(self):
        """
        Compile the Gradient Boosted Trees model.
        """
        logger.info("Compiling Gradient Boosted Trees model...")

        # Set a dummy loss function and optimizer, as we will train the XGBoost model directly
        self.model.compile(loss='mse', optimizer='adam')

        logger.info("Gradient Boosted Trees model compiled.")

    def train_model(self, X_train, y_train):
        """
        Train the Gradient Boosted Trees model on the provided training data.

        Args:
            X_train (numpy.ndarray): Training input features.
            y_train (numpy.ndarray): Training target labels.
        """
        logger.info("Training Gradient Boosted Trees model...")

        # Train the XGBoost model directly
        self.xgb_model.fit(X_train, y_train)

        logger.info("Gradient Boosted Trees model training completed.")

    def save_model(self, output_dir):
        """
        Save the trained Gradient Boosted Trees model to a file.

        Args:
            output_dir (str): Directory to save the model file.
        """
        # Ensure the target directory exists
        expanded_output_dir = os.path.expanduser(output_dir)
        try:
            os.makedirs(expanded_output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directories: {e}")

        model_filename = f"model.xgb"
        model_path = os.path.join(output_dir, model_filename)
        self.xgb_model.save_model(model_path)

        # Create a companion file with model configuration details
        companion_filename = f"hyperparameters.json"
        companion_path = os.path.join(output_dir, companion_filename)
        model_config = {
            "max_depth": self.xgb_model.max_depth,
            "learning_rate": self.xgb_model.learning_rate,
            "n_estimators": self.xgb_model.n_estimators,
            "objective": self.xgb_model.objective,
            "num_class": self.xgb_model.num_class,
            "tree_method": self.xgb_model.tree_method,
        }
        with open(companion_path, "w") as file:
            json.dump(model_config, file, indent=4)

        logger.info(f"Gradient Boosted Trees model saved to {model_path}")
        logger.info(f"Model configuration saved to {companion_path}")
