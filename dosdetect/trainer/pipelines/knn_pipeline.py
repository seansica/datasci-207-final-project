# pipelines/knn_pipeline.py
import json
import os

from ..utils.evaluation import SKLearnEvaluator

from .base_pipeline import BasePipeline
from ..models.knn import KNN
from ..utils.logger import init_logger

logger = init_logger(__name__)


class KNNPipeline(BasePipeline):

    def __init__(
        self,
        dataset_file_paths,
        pipeline_dir,
        auto_tune=True,
        train_fraction=1.0,
        correlation_threshold=None,
        pca_variance_ratio=None,
        n_neighbors=None,
    ):
        super().__init__(
            dataset_file_paths,
            pipeline_dir,
            auto_tune,
            train_fraction,
            correlation_threshold,
            pca_variance_ratio,
        )
        self.n_neighbors = n_neighbors

    def run(self):
        logger.info("Starting KNN pipeline...")

        pipeline_details = {
            "pipeline_type": "KNN",
            "auto_tune": self.auto_tune,
            "train_fraction": self.train_fraction,
            "correlation_threshold": self.correlation_threshold,
            "pca_variance_ratio": self.pca_variance_ratio,
            "n_neighbors": self.n_neighbors,
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

        knn = KNN(n_neighbors=self.n_neighbors, auto_tune=self.auto_tune)
        knn.build_model()
        logger.info("KNN model initialized.")

        knn.train(X_train, y_train)
        logger.info("KNN model trained.")

        knn.save_model(self.pipeline_dir)
        logger.info("KNN model saved.")

        knn.save_model(self.pipeline_dir)
        logger.info("Model saved.")

        evaluator = SKLearnEvaluator(knn.model, self.pipeline_dir, label_mappings)
        evaluator.evaluate(X_test, y_test)

        logger.info("KNN pipeline finished.")
