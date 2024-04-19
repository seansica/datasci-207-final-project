import json
import os

from .base_pipeline import BasePipeline
from ..models.gradient_boosted_trees import GradientBoostedTreesModel
from ..preprocessing.preprocessor import PreprocessorBuilder
from ..utils.evaluation import SKLearnEvaluator
from ..utils.logger import init_logger

logger = init_logger(__name__)


class GradientBoostedTreesPipeline(BasePipeline):
    def __init__(
        self,
        dataset_file_paths,
        pipeline_dir,
        auto_tune=True,
        train_fraction=1.0,
        correlation_threshold=None,
        pca_variance_ratio=None,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
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
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators

    def run(self):
        logger.info("Starting Gradient Boosted Trees pipeline...")

        pipeline_details = {
            "pipeline_type": "Gradient Boosted Trees",
            "auto_tune": self.auto_tune,
            "train_fraction": self.train_fraction,
            "correlation_threshold": self.correlation_threshold,
            "pca_variance_ratio": self.pca_variance_ratio,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
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

        # Determine the input shape
        input_shape = X_train.shape[1:]

        # Initialize the model
        gbt_model = GradientBoostedTreesModel(input_shape, num_classes, self.auto_tune)
        gbt_model.build_model(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
        )

        logger.info("Gradient Boosted Trees model created.")

        # Compile the model (dummy compilation for Keras wrapper)
        gbt_model.compile_model()

        logger.debug("Model compiled.")

        logger.info("Training the model...")

        # Train the model
        gbt_model.train_model(X_train, y_train)

        logger.info("Model training completed.")

        # Save the model
        gbt_model.save_model(self.pipeline_dir)

        logger.info("Model saved.")

        # Evaluate the model
        evaluator = SKLearnEvaluator(
            gbt_model.xgb_model, self.pipeline_dir, label_mappings
        )
        evaluator.evaluate(X_test, y_test)

        logger.info("Gradient Boosted Trees pipeline finished.")
