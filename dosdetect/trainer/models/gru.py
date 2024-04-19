import os
import json
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

from ..utils.logger import init_logger

logger = init_logger(__name__)


class GRUModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        logger.debug(
            f"GRU model initialized with input shape: {input_shape}, num_classes: {num_classes}"
        )

    def build_model(self, units=128, dropout=0.2):
        logger.info("Building GRU model...")

        model = Sequential()
        model.add(GRU(units, input_shape=self.input_shape, return_sequences=True))
        model.add(GRU(units))
        model.add(Dense(self.num_classes, activation="softmax"))
        self.model = model

        logger.debug(f"GRU model architecture: units: {units}, dropout: {dropout}")
        logger.info("GRU model built.")

    def compile_model(
        self, optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    ):
        logger.info("Compiling GRU model...")

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        logger.info("GRU model compiled.")

    def train_model(self, X_train, y_train, epochs, batch_size, validation_data=None):
        logger.info(f"Training GRU model for {epochs} epochs...")

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
        )

        logger.info("GRU model training completed.")
        return history

    def save_model(self, output_dir):
        expanded_output_dir = os.path.expanduser(output_dir)
        try:
            os.makedirs(expanded_output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directories: {e}")

        model_path = os.path.join(output_dir, "model.h5")
        self.model.save(model_path)

        companion_path = os.path.join(output_dir, "hyperparameters.json")
        model_config = {
            "units": self.model.layers[0].units,
            "dropout": self.model.layers[0].dropout,
            "optimizer": self.model.optimizer.get_config(),
            "loss": self.model.loss,
            "metrics": self.model.metrics_names,
        }
        with open(companion_path, "w") as file:
            json.dump(model_config, file, indent=4, default=lambda x: str(x))

        logger.info(f"GRU model saved to {model_path}")
        logger.info(f"Model configuration saved to {companion_path}")
