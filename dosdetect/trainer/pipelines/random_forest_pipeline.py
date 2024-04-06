# pipelines/random_forest_pipeline.py
import json
import os

from .base_pipeline import BasePipeline
from ..models.random_forest import RandomForest
from ..utils.logger import init_logger

logger = init_logger("random_forest_pipeline_logger")


class RandomForestPipeline(BasePipeline):
    def __init__(
        self,
        dataset_file_paths,
        pipeline_dir,
        correlation_threshold,
        pca_variance_ratio,
        n_estimators,
        max_depth,
        random_state,
    ):
        super().__init__(
            dataset_file_paths, pipeline_dir, correlation_threshold, pca_variance_ratio
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def run(self):
        logger.info("Starting Random Forest pipeline...")

        pipeline_details = {
            "pipeline_type": "RandomForest",
            "correlation_threshold": self.correlation_threshold,
            "pca_variance_ratio": self.pca_variance_ratio,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
        }

        with open(os.path.join(self.pipeline_dir, "pipeline_details.json"), "w") as f:
            json.dump(pipeline_details, f)

        data_loader, X_preprocessed, y_encoded, label_mappings = self.preprocess_data()

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data(
            X_preprocessed, y_encoded
        )

        logger.debug(
            f"Data split into train, validation, and test sets. "
            f"Train: {X_train.shape}, {y_train.shape}, "
            f"Validation: {X_val.shape}, {y_val.shape}, "
            f"Test: {X_test.shape}, {y_test.shape}"
        )

        random_forest = RandomForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        random_forest.build_model()
        logger.info("Random Forest model initialized.")

        random_forest.train(X_train, y_train)
        logger.info("Random Forest model trained.")

        random_forest.save_model(self.pipeline_dir)
        logger.info("Random Forest model saved.")

        self.evaluate_model(
            random_forest.model, self.pipeline_dir, X_test, y_test, label_mappings
        )

        logger.info("Random Forest pipeline finished.")
