import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np
import os
import json

from ..utils.logger import init_logger

logger = init_logger("evaluator_logger")


class Evaluator:
    """
    A class for evaluating a trained model.
    """

    def __init__(self, model, output_dir, base_filename, label_mappings):
        """
        Initialize the Evaluator with a trained model.

        Args:
            model (tensorflow.keras.Model or sklearn.base.BaseEstimator): The trained model to evaluate.
            output_dir (str): Directory to save evaluation results.
            base_filename (str): Base filename for saving evaluation plots and metrics.
            label_mappings (dict): The mappings of encoded labels to original labels.
        """
        self.model = model
        self.label_mappings = label_mappings
        self.output_dir = output_dir
        self.base_filename = base_filename

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        logger.debug("Evaluator initialized with the trained model.")

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set.

        Args:
            X_test (numpy.ndarray): Test input features.
            y_test (numpy.ndarray): Test target labels.

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        logger.debug(f"Evaluating model of type: {type(self.model)}")

        if isinstance(self.model, tf.keras.models.Model):
            metrics = self._evaluate_keras_model(X_test, y_test)
        else:
            metrics = self._evaluate_sklearn_model(X_test, y_test)

        self._save_evaluation_metrics(metrics)
        return metrics

    def _evaluate_keras_model(self, X_test, y_test):
        """
        Evaluate a Keras model.
        """
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        return self._compute_metrics(y_test, y_pred, y_pred_prob)

    def _evaluate_sklearn_model(self, X_test, y_test):
        """
        Evaluate a scikit-learn model.
        """
        y_pred = self.model.predict(X_test)
        y_pred_prob = self.model.predict_proba(X_test)
        return self._compute_metrics(y_test, y_pred, y_pred_prob)

    def _compute_metrics(self, y_true, y_pred, y_prob):
        """
        Compute evaluation metrics for each class label.
        """
        metrics = {}
        for encoded_label, label_name in self.label_mappings.items():
            logger.info(f"Computing metrics for class: {label_name}")
            metrics[label_name] = {
                "accuracy": accuracy_score(y_true == encoded_label, y_pred == encoded_label),
                "precision": precision_score(y_true == encoded_label, y_pred == encoded_label),
                "recall": recall_score(y_true == encoded_label, y_pred == encoded_label),
                "f1_score": f1_score(y_true == encoded_label, y_pred == encoded_label),
            }
            self._plot_roc_curve(y_true == encoded_label, y_prob[:, encoded_label], label_name)
            self._plot_confusion_matrix(y_true == encoded_label, y_pred == encoded_label, label_name)

        return metrics

    def _plot_roc_curve(self, y_true, y_prob, label_name):
        """
        Plot the ROC curve for a specific class label and save it to a file.
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"ROC Curve - {label_name}")
        plt.legend(loc="lower right")

        filename = f"{self.base_filename}_{label_name}_roc_curve.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def _plot_confusion_matrix(self, y_true, y_pred, label_name):
        """
        Plot the confusion matrix for a specific class label and save it to a file.
        """
        matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {label_name}")
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        plt.tight_layout()

        filename = f"{self.base_filename}_{label_name}_confusion_matrix.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def _save_evaluation_metrics(self, metrics):
        """
        Save the evaluation metrics to a JSON file.
        """
        filename = f"{self.base_filename}_evaluation_metrics.json"
        with open(os.path.join(self.output_dir, filename), "w") as file:
            json.dump(metrics, file, indent=4)
