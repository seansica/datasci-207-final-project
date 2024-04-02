# models/knn.py
from sklearn.neighbors import KNeighborsClassifier
import logging
from ..utils.logger import setup_logger

logger = setup_logger('knn_logger', 'knn.log', level=logging.DEBUG)


class KNN:
    """
    A class for creating and training a K-Nearest Neighbors (KNN) model.
    """

    def __init__(self, n_neighbors=5):
        """
        Initialize the KNN model with the specified number of neighbors.

        Args:
            n_neighbors (int): Number of neighbors to consider. Defaults to 5.
        """
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        logger.debug(f"KNN initialized with n_neighbors={n_neighbors}")

    def train(self, X_train, y_train):
        """
        Train the KNN model on the provided training data.

        Args:
            X_train (numpy.ndarray): Training input features.
            y_train (numpy.ndarray): Training target labels.
        """
        logger.info("Training KNN model...")
        self.model.fit(X_train, y_train)
        logger.info("KNN model training completed.")

    def predict(self, X):
        """
        Make predictions using the trained KNN model.

        Args:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        logger.info("Making predictions with KNN model...")
        predictions = self.model.predict(X)
        logger.info("Predictions completed.")
        return predictions