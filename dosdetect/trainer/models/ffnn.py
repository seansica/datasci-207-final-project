import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from ..utils.logger import init_logger

logger = init_logger(__name__)


class FFNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        logger.debug(
            f"FFNN model initialized with input shape: {input_shape}, num_classes: {num_classes}"
        )

    def build_model(self, hidden_units=128, dropout_rate=0.2, num_hidden_layers=2):
        logger.info("Building FFNN model...")

        model = Sequential()
        model.add(Dense(hidden_units, input_shape=self.input_shape, activation="relu"))
        model.add(Dropout(dropout_rate))

        for _ in range(num_hidden_layers - 1):
            model.add(Dense(hidden_units, activation="relu"))
            model.add(Dropout(dropout_rate))

        model.add(Dense(self.num_classes, activation="softmax"))
        self.model = model

        logger.debug(
            f"FFNN model architecture: hidden_units: {hidden_units}, dropout_rate: {dropout_rate}, num_hidden_layers: {num_hidden_layers}"
        )
        logger.info("FFNN model built.")

    def compile_model(
        self, optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    ):
        logger.info("Compiling FFNN model...")

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        logger.info("FFNN model compiled.")

    def train_model(self, X_train, y_train, epochs, batch_size, validation_data=None):
        logger.info(f"Training FFNN model for {epochs} epochs...")

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
        )

        logger.info("FFNN model training completed.")
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
            "hidden_units": self.model.layers[0].units,
            "dropout_rate": self.model.layers[1].rate,
            "num_hidden_layers": len(
                [layer for layer in self.model.layers if isinstance(layer, Dense)]
            )
            - 1,
            "optimizer": self.model.optimizer.get_config(),
            "loss": self.model.loss,
            "metrics": self.model.metrics_names,
        }
        with open(companion_path, "w") as file:
            json.dump(model_config, file, indent=4, default=lambda x: str(x))

        logger.info(f"FFNN model saved to {model_path}")
        logger.info(f"Model configuration saved to {companion_path}")
