import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import numpy as np
import os

from ..utils.logger import setup_logger

logger = setup_logger('evaluator_logger', 'evaluator.log', level=logging.DEBUG)


class Evaluator:
    """
    A class for evaluating a trained model.
    """

    def __init__(self, model, output_dir, base_filename):
        """
        Initialize the Evaluator with a trained model.

        Args:
            model (tensorflow.keras.Model): The trained model to evaluate.
            output_dir (str): Directory to save evaluation results.
            base_filename (str): Base filename for saving evaluation plots.
        """
        self.model = model
        self.output_dir = output_dir
        self.base_filename = base_filename
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        logger.debug("Evaluator initialized with the trained model.")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set. Automatically detects if the model
        is a Keras model or a scikit-learn model and applies the appropriate 
        evaluation method.

        Args:
            X_test (numpy.ndarray): Test input features.
            y_test (numpy.ndarray): Test target labels.

        Returns:
            If Keras model: Returns the loss and accuracy.
            If scikit-learn model: Returns the accuracy.
        """
        logger.debug(f"Evaluating model of type: {type(self.model)}")

        if isinstance(self.model, tf.keras.models.Model):
            # Keras model evaluation
            loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"Keras Model - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
            metrics = {'loss': loss, 'accuracy': accuracy}
        else:
            # Scikit-learn model evaluation
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Scikit-learn Model - Test Accuracy: {accuracy:.4f}")
            metrics = {'accuracy': accuracy}
        
        self.plot_confusion_matrix(y_test, y_pred, title='Confusion Matrix')
        
        if hasattr(self.model, 'predict_proba') and len(np.unique(y_test)) == 2:
            # Only plot ROC curve for binary classification problems with probability output
            y_prob = self.model.predict_proba(X_test)[:, 1]
            self.plot_roc_curve(y_test, y_prob, title='ROC Curve')
        
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, title):
        """
        Plot the confusion matrix and save it to a file.
        """
        matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        filename = f"{self.base_filename}_confusion_matrix.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plot_roc_curve(self, y_true, y_prob, title):
        """
        Plot the ROC curve and save it to a file.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        filename = f"{self.base_filename}_roc_curve.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()