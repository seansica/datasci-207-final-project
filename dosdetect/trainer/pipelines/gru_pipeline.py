import json
import os
from .base_pipeline import BasePipeline
from ..models.gru import GRUModel
from ..preprocessing.preprocessor import PreprocessorBuilder
from ..utils.evaluation import KerasEvaluator
from ..utils.logger import init_logger

logger = init_logger(__name__)


class GRUPipeline(BasePipeline):
    def __init__(
        self,
        dataset_file_paths,
        pipeline_dir,
        auto_tune=True,
        train_fraction=1.0,
        correlation_threshold=None,
        pca_variance_ratio=None,
        epochs=10,
        batch_size=32,
    ):
        super().__init__(
            dataset_file_paths,
            pipeline_dir,
            auto_tune,
            train_fraction,
            correlation_threshold,
            pca_variance_ratio,
        )
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        logger.info("Starting GRU pipeline...")

        pipeline_details = {
            "pipeline_type": "GRU",
            "auto_tune": self.auto_tune,
            "train_fraction": self.train_fraction,
            "correlation_threshold": self.correlation_threshold,
            "pca_variance_ratio": self.pca_variance_ratio,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
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
        input_shape = (X_train.shape[1], 1)

        # Initialize the model
        gru_model = GRUModel(input_shape, num_classes)
        gru_model.build_model()

        logger.info("GRU model created.")

        # Compile the model
        gru_model.compile_model(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        logger.debug(
            "Model compiled with Adam optimizer, categorical cross-entropy loss, and accuracy metric."
        )

        logger.info(f"Training the model for {self.epochs} epochs...")

        # Train the model
        history = gru_model.train_model(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
        )

        logger.info("Model training completed.")

        # Save the model
        gru_model.save_model(self.pipeline_dir)

        logger.info("Model saved.")

        # Evaluate the model
        evaluator = KerasEvaluator(gru_model.model, self.pipeline_dir, label_mappings)
        evaluator.evaluate(X_test, y_test, history)

        logger.info("GRU pipeline finished.")
