import os
import json
import logging
from sklearn.tree import DecisionTreeClassifier
import joblib

from ..utils.logger import init_logger

logger = init_logger(__name__)


class DecisionTreeModel:
    def __init__(
        self,
        num_classes,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
    ):
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.model = None
        logger.debug(
            f"Decision Tree model initialized with num_classes: {num_classes}, max_depth: {max_depth}, "
            f"min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, criterion: {criterion}"
        )

    def build_model(self):
        logger.info("Building Decision Tree model...")

        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
        )

        logger.info("Decision Tree model built.")

    def train_model(self, X_train, y_train):
        logger.info("Training Decision Tree model...")

        self.model.fit(X_train, y_train)

        logger.info("Decision Tree model training completed.")

    def save_model(self, output_dir):
        expanded_output_dir = os.path.expanduser(output_dir)
        try:
            os.makedirs(expanded_output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directories: {e}")

        model_path = os.path.join(output_dir, "model.joblib")
        joblib.dump(self.model, model_path)

        companion_path = os.path.join(output_dir, "hyperparameters.json")
        model_config = {
            "max_depth": self.model.max_depth,
            "min_samples_split": self.model.min_samples_split,
            "min_samples_leaf": self.model.min_samples_leaf,
            "criterion": self.model.criterion,
        }
        with open(companion_path, "w") as file:
            json.dump(model_config, file, indent=4)

        logger.info(f"Decision Tree model saved to {model_path}")
        logger.info(f"Model configuration saved to {companion_path}")
