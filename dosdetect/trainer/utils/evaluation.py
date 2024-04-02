import logging
from ..utils.logger import setup_logger

logger = setup_logger('evaluator_logger', 'evaluator.log', level=logging.DEBUG)


class Evaluator:
    """
    A class for evaluating a trained model.
    """

    def __init__(self, model):
        """
        Initialize the Evaluator with a trained model.

        Args:
            model (tensorflow.keras.Model): The trained model to evaluate.
        """
        self.model = model
        logger.debug("Evaluator initialized with the trained model.")

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data.

        Args:
            X_test (numpy.ndarray): Test input features.
            y_test (numpy.ndarray): Test target labels.

        Returns:
            tuple: A tuple containing the test loss and test accuracy.
        """
        logger.info("Evaluating the model on the test data...")

        # Evaluate the model on the test data
        loss, accuracy = self.model.evaluate(X_test, y_test)

        logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        # Log the evaluation metrics
        logger.debug(f"Evaluation Metrics:")
        logger.debug(f"  Test Loss: {loss:.4f}")
        logger.debug(f"  Test Accuracy: {accuracy:.4f}")

        return loss, accuracy