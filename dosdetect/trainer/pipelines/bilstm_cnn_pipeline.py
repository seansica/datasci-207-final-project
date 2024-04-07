# pipelines/bilstm_cnn_pipeline.py
from datetime import datetime
import json
import os
import logging

from .base_pipeline import BasePipeline
from ..models.bilstm_cnn import BiLSTMCNN
from ..utils.logger import init_logger

logger = init_logger("bilstm_cnn_pipeline_logger")


class BiLSTMCNNPipeline(BasePipeline):

    def __init__(
        self,
        dataset_file_paths,
        pipeline_dir,
        auto_tune=True,
        correlation_threshold=None,
        pca_variance_ratio=None,
        epochs=None,
        batch_size=None,
    ):
        super().__init__(
            dataset_file_paths,
            pipeline_dir,
            auto_tune,
            correlation_threshold,
            pca_variance_ratio,
        )
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        logger.info("Starting BiLSTM-CNN pipeline...")

        pipeline_details = {
            "pipeline_type": "BiLSTM-CNN",
            "auto_tune": self.auto_tune,
            "correlation_threshold": self.correlation_threshold,
            "pca_variance_ratio": self.pca_variance_ratio,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }

        with open(os.path.join(self.pipeline_dir, "pipeline_details.json"), "w") as f:
            json.dump(pipeline_details, f)

        data_loader, X_preprocessed, y_encoded, label_mappings = self.preprocess_data()

        num_classes = y_encoded.shape[1]
        logger.debug(f"Number of classes: {num_classes}")

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data(X_preprocessed, y_encoded)
        logger.debug(f"Data split into train, validation, and test sets. "
                     f"Train: {X_train.shape}, {y_train.shape}, "
                     f"Validation: {X_val.shape}, {y_val.shape}, "
                     f"Test: {X_test.shape}, {y_test.shape}")

        input_shape = (X_train.shape[1], 1)
        bilstm_cnn = BiLSTMCNN(input_shape, num_classes, self.auto_tune)
        bilstm_cnn.build_model()
        logger.info("BiLSTM-CNN model created.")

        bilstm_cnn.compile_model(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.debug("Model compiled with Adam optimizer, categorical cross-entropy loss, and accuracy metric.")

        logger.info(f"Training the model for {self.epochs} epochs...")
        bilstm_cnn.train_model(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
        )
        logger.info("Model training completed.")

        # test_loss, test_accuracy = bilstm_cnn.evaluate_model(X_test, y_test)
        # logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        bilstm_cnn.save_model(self.pipeline_dir)
        logger.info("Model saved.")

        self.evaluate_model(
            bilstm_cnn.model,
            self.pipeline_dir,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            label_mappings,
        )

        logger.info("BiLSTM-CNN pipeline finished.")
