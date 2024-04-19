from datetime import datetime
import json
import os
import logging

from .base_pipeline import BasePipeline
from ..models.ffnn import FFNN
from ..preprocessing.preprocessor import PreprocessorBuilder
from ..utils.evaluation import KerasEvaluator
from ..utils.logger import init_logger

logger = init_logger(__name__)


class FFNNPipeline(BasePipeline):
    def __init__(
        self,
        dataset_file_paths,
        pipeline_dir,
        auto_tune=True,
        train_fraction=1.0,
        correlation_threshold=None,
        pca_variance_ratio=None,
        hidden_units=128,
        dropout_rate=0.2,
        num_hidden_layers=2,
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
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        logger.info("Starting FFNN pipeline...")

        pipeline_details = {
            "pipeline_type": "FFNN",
            "auto_tune": self.auto_tune,
            "train_fraction": self.train_fraction,
            "correlation_threshold": self.correlation_threshold,
            "pca_variance_ratio": self.pca_variance_ratio,
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
            "num_hidden_layers": self.num_hidden_layers,
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
        input_shape = X_train.shape[1:]

        # Initialize the model
        ffnn_model = FFNN(input_shape, num_classes)
        ffnn_model.build_model(
            hidden_units=self.hidden_units,
            dropout_rate=self.dropout_rate,
            num_hidden_layers=self.num_hidden_layers,
        )

        logger.info("FFNN model created.")

        # Compile the model
        ffnn_model.compile_model()

        logger.debug("Model compiled.")

        logger.info(f"Training the model for {self.epochs} epochs...")

        # Train the model
        history = ffnn_model.train_model(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
        )

        logger.info("Model training completed.")

        # Save the model
        ffnn_model.save_model(self.pipeline_dir)

        logger.info("Model saved.")

        # Evaluate the model
        evaluator = KerasEvaluator(ffnn_model.model, self.pipeline_dir, label_mappings)
        evaluator.evaluate(X_test, y_test, history)

        logger.info("FFNN pipeline finished.")
