from datetime import datetime
import os
import json
import logging
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Conv1D, Dense, Dropout

from ..utils.logger import init_logger

logger = init_logger("bilstm_cnn_logger")


class BiLSTMCNN:
    """
    A class for creating and managing a Bidirectional LSTM-CNN model.
    """

    def __init__(self, input_shape, num_classes):
        """
        Initialize the BiLSTMCNN model with input shape and number of classes.

        Args:
            input_shape (tuple): Shape of the input data.
            num_classes (int): Number of output classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        logger.debug(f"BiLSTMCNN initialized with input shape: {input_shape}, num_classes: {num_classes}")

    def build_model(self):
        """
        Build the BiLSTM-CNN model architecture.
        """
        logger.info("Building BiLSTM-CNN model...")

        # Create the model architecture
        self.model = tf.keras.Sequential([
            # Convolutional layer
            Conv1D(64, 3, activation='relu', input_shape=self.input_shape),
            
            # Bidirectional LSTM layers
            Bidirectional(LSTM(128, return_sequences=True)),
            Bidirectional(LSTM(64)),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

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
            y_train (numpy.ndarray): Training target labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            validation_data (tuple): Tuple of (X_val, y_val) for validation during training.
        """
        logger.info(f"Training BiLSTM-CNN model for {epochs} epochs...")

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

        logger.info("BiLSTM-CNN model training completed.")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained BiLSTM-CNN model on the provided test data.

        Args:
            X_test (numpy.ndarray): Test input features.
            y_test (numpy.ndarray): Test target labels.

        Returns:
            tuple: A tuple containing the test loss and test accuracy.
        """
        logger.info("Evaluating BiLSTM-CNN model...")

        loss, accuracy = self.model.evaluate(X_test, y_test)

        logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        return loss, accuracy

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
            "optimizer": self.optimizer,
            "loss": self.loss,
            "metrics": self.metrics,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }
        with open(companion_path, "w") as file:
            json.dump(model_config, file, indent=4)

        logger.info(f"BiLSTM-CNN model saved to {model_path}")
        logger.info(f"Model configuration saved to {companion_path}")
