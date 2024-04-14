from datetime import datetime
import os
import json
import logging
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Attention
from kerastuner import HyperParameters, RandomSearch

from ..utils.logger import init_logger

logger = init_logger(__name__)


class BiLSTMCNN:
    """
    A class for creating and managing a Bidirectional LSTM-CNN model.
    """

    def __init__(self, input_shape, num_classes, auto_tune=True):
        """
        Initialize the BiLSTMCNN model with input shape and number of classes.

        Args:
            input_shape (tuple): Shape of the input data.
            num_classes (int): Number of output classes.
            auto_tune (bool): Whether to use hyperparameter autotuning. Defaults to True.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.auto_tune = auto_tune
        self.model = None
        logger.debug(
            f"BiLSTMCNN initialized with input shape: {input_shape}, num_classes: {num_classes}, auto_tune: {auto_tune}"
        )

    def build_model(self, hp=None, binary_classification=False):
        """
        Build the BiLSTM-CNN model architecture.

        Args:
            hp (HyperParameters): Hyperparameters for tuning. Defaults to None.
            binary_classification (bool): Whether to use binary classification or multi-class classification.
        """
        logger.info("Building BiLSTM-CNN model with attention mechanism...")

        # Create the model architecture
        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.InputLayer(shape=self.input_shape))

        # TODO Attention layer
        # model.add(Attention(use_scale=True))

        # Bidirectional LSTM layers
        model.add(
            Bidirectional(
                LSTM(
                    units=(
                        hp.Int("lstm_units_1", min_value=64, max_value=256, step=64)
                        if hp
                        else 128
                    ),
                    return_sequences=True,
                )
            )
        )
        model.add(
            Bidirectional(
                LSTM(
                    units=(
                        hp.Int("lstm_units_2", min_value=32, max_value=128, step=32)
                        if hp
                        else 64
                    ),
                    # The output shape of the last LSTM layer is (batch_size, units), where units is the number of units in the last LSTM layer.
                    # However, the Conv1D layer expects an input shape of (batch_size, timesteps, features), where timesteps is the number of
                    # time steps and features is the number of features per time step.
                    # To resolve this issue, you can modify the last LSTM layer to return sequences by setting return_sequences=True.
                    # This will ensure that the output shape of the last LSTM layer is (batch_size, timesteps, units), which is compatible with
                    # the input shape expected by the Conv1D layer.
                    return_sequences=True,
                )
            )
        )

        # Convolutional layer
        model.add(
            Conv1D(
                filters=(
                    hp.Int("conv_filters", min_value=32, max_value=128, step=32)
                    if hp
                    else 64
                ),
                kernel_size=3,
                activation="relu",
            )
        )

        # Flatten layer
        model.add(Flatten())

        # Dense layers
        model.add(
            Dense(
                units=(
                    hp.Int("dense_units", min_value=32, max_value=128, step=32)
                    if hp
                    else 64
                ),
                activation="relu",
            )
        )
        model.add(
            Dropout(
                rate=(
                    hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
                    if hp
                    else 0.5
                )
            )
        )

        # Output layer
        num_output_units = 1 if binary_classification else self.num_classes
        activation = "sigmoid" if binary_classification else "softmax"
        model.add(Dense(num_output_units, activation=activation))

        self.model = model

        logger.debug("BiLSTM-CNN model architecture:")
        self.model.summary(print_fn=lambda x, **kwargs: logger.debug(x))

        logger.info("BiLSTM-CNN model built.")

    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        """
        Compile the BiLSTM-CNN model with the specified optimizer, loss function, and metrics.

        Args:
            optimizer (str or tf.keras.optimizers.Optimizer): The optimizer to use for training.
            loss (str or tf.keras.losses.Loss): The loss function to use for training.
            metrics (list): The metrics to evaluate during training.
        """
        logger.info("Compiling BiLSTM-CNN model...")

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        logger.info("BiLSTM-CNN model compiled.")

    def train_model(self, X_train, y_train, epochs, batch_size, validation_data=None):
        """
        Train the BiLSTM-CNN model on the provided training data.

        Args:
            X_train (numpy.ndarray): Training input features.
            y_train (numpy.ndarray): Training target labels (already one-hot encoded).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            validation_data (tuple): Tuple of (X_val, y_val) for validation during training.
                                    y_val should be already one-hot encoded.

        Returns:
            tf.keras.callbacks.History: The history object containing training details.
        """
        logger.info(f"Training BiLSTM-CNN model for {epochs} epochs...")

        # Create early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping],
        )

        logger.info("BiLSTM-CNN model training completed.")
        return history

    def save_model(self, output_dir):
        """
        Save the trained BiLSTM-CNN model to a file.

        Args:
            output_dir (str): Directory to save the model file.
        """
        # Ensure the target directory exists
        expanded_output_dir = os.path.expanduser(output_dir)
        try:
            os.makedirs(expanded_output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directories: {e}")

        model_filename = f"model.keras"
        model_path = os.path.join(output_dir, model_filename)
        self.model.save(model_path)

        # Create a companion file with model configuration details
        companion_filename = f"hyperparameters.json"
        companion_path = os.path.join(output_dir, companion_filename)
        model_config = {
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "optimizer": self.model.optimizer.get_config(),
            "loss": self.model.loss,
            "metrics": self.model.metrics_names,
        }
        with open(companion_path, "w") as file:
            json.dump(model_config, file, indent=4)

        logger.info(f"BiLSTM-CNN model saved to {model_path}")
        logger.info(f"Model configuration saved to {companion_path}")
