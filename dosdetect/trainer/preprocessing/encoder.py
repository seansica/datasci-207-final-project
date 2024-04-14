from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd

from ..utils.logger import init_logger

logger = init_logger(__name__)


class LabelEncoderHandler:
    def __init__(self, mode="sparse"):
        self.label_encoder = None
        self.mode = mode

    def encode_labels(self, y):
        """
        Encode target labels using either Label Encoder or One-Hot Encoder, returning label mappings for both.

        Args:
            y (numpy.ndarray or pandas.Series): The target labels.
            mode (str): The mode of encoding, 'sparse' for Label Encoder and 'one_hot' for One-Hot Encoding.

        Returns:
            numpy.ndarray: The encoded labels, either as integers or one-hot encoded vectors.
            dict: Mapping of original labels to encoded values or encoded indices.
        """
        if self.mode == "sparse":
            if isinstance(y, np.ndarray):
                y = pd.Series(y)
            self.label_encoder = LabelEncoder()
            logger.debug("Encoding labels with Label Encoder...")
            y_encoded = self.label_encoder.fit_transform(y)
            label_mapping = dict(
                zip(
                    self.label_encoder.classes_,
                    self.label_encoder.transform(self.label_encoder.classes_),
                )
            )
            logger.debug(f"Label Mapping: {label_mapping}")
            logger.debug("Label encoding completed.")
            return y_encoded.ravel(), label_mapping

        elif self.mode == "one_hot":
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            num_classes = len(self.label_encoder.classes_)
            logger.debug("One-hot encoding labels...")
            y_one_hot = to_categorical(y_encoded, num_classes=num_classes)
            label_mapping = {
                cls: idx for idx, cls in enumerate(self.label_encoder.classes_)
            }
            logger.debug(
                f"One-hot encoding completed. Number of classes: {num_classes}"
            )
            logger.debug(f"Label Mapping for One-Hot: {label_mapping}")
            return y_one_hot, label_mapping

        else:
            raise ValueError("Invalid mode specified. Use 'label' or 'one_hot'.")
