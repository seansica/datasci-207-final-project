from datetime import datetime
import json
import os
import logging

from .base_pipeline import BasePipeline
from ..models.decision_tree import DecisionTreeModel
from ..preprocessing.preprocessor import PreprocessorBuilder
from ..utils.evaluation import SKLearnEvaluator
from ..utils.logger import init_logger

logger = init_logger(__name__)


class DecisionTreePipeline(BasePipeline):
    def __init__(
        self,
        dataset_file_paths,
        pipeline_dir,
        auto_tune=True,
        train_fraction=1.0,
        correlation_threshold=None,
        pca_variance_ratio=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
    ):
        super().__init__(
            dataset_file_paths,
            pipeline_dir,
            auto_tune,
            train_fraction,
            correlation_threshold,
            pca_variance_ratio,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion

    def run(self):
        logger.info("Starting Decision Tree pipeline...")

        pipeline_details = {
            "pipeline_type": "Decision Tree",
            "auto_tune": self.auto_tune,
            "train_fraction": self.train_fraction,
            "correlation_threshold": self.correlation_threshold,
            "pca_variance_ratio": self.pca_variance_ratio,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion,
        }

        with open(os.path.join(self.pipeline_dir, "pipeline_details.json"), "w") as f:
            json.dump(pipeline_details, f)

        preprocessor = (
            PreprocessorBuilder()
            .with_data_cleaning(fill_method="median")
            .with_correlated_feature_removal(
                correlation_threshold=self.correlation_threshold
            )
            .with_pca(pca_variance_ratio=self.pca_variance_ratio)
            .with_one_hot_encoding()
            .build()
        )

        # Load and process the data
        data_loader, X_preprocessed, y_encoded, label_mappings = self.initialize(
            preprocessor=preprocessor
        )

        num_classes = len(label_mappings)
        logger.debug(f"Number of classes: {num_classes}")

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data(
            X_preprocessed, y_encoded
        )

        logger.debug(
            f"Data split into train, validation, and test sets. "
            f"Train: {X_train.shape}, {y_train.shape}, "
            f"Validation: {X_val.shape}, {y_val.shape}, "
            f"Test: {X_test.shape}, {y_test.shape}"
        )

        # Initialize the model
        dt_model = DecisionTreeModel(
            num_classes,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
        )
        dt_model.build_model()

        logger.info("Decision Tree model created.")

        logger.info("Training the model...")

        # Train the model
        dt_model.train_model(X_train, y_train)

        logger.info("Model training completed.")

        # Save the model
        dt_model.save_model(self.pipeline_dir)

        logger.info("Model saved.")

        # Evaluate the model
        evaluator = SKLearnEvaluator(dt_model.model, self.pipeline_dir, label_mappings)
        evaluator.evaluate(X_test, y_test)

        logger.info("Decision Tree pipeline finished.")
