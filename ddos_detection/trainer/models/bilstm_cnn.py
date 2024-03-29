import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Conv1D, Dense, Dropout

import logging
from ..utils.logger import setup_logger

logger = setup_logger('bilstm_cnn_logger', 'bilstm_cnn.log', level=logging.DEBUG)


class BiLSTMCNN:
    """
    A class for creating a Bidirectional LSTM-CNN model.
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
        logger.debug(f"BiLSTMCNN initialized with input shape: {input_shape}, num_classes: {num_classes}")

    def create_model(self):
        """
        Create and return the BiLSTM-CNN model.

        Returns:
            tensorflow.keras.Model: The created BiLSTM-CNN model.
        """
        logger.info("Creating BiLSTM-CNN model...")

        # Create the model architecture
        model = tf.keras.Sequential([
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
        model.summary(print_fn=lambda x: logger.debug(x))

        logger.info("BiLSTM-CNN model created.")

        return model