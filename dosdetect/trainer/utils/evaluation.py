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

    def __init__(self, model, output_dir, label_mappings):
        """
        Initialize the Evaluator with a trained model.

        Args:
            model (tensorflow.keras.Model or sklearn.base.BaseEstimator): The trained model to evaluate.
            output_dir (str): Directory to save evaluation results.
            label_mappings (dict): The mappings of encoded labels to original labels.
        """
        self.model = model
        self.label_mappings = label_mappings
        self.output_dir = os.path.expanduser(output_dir)

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
        for label_name, encoded_label in self.label_mappings.items():
            logger.info(f"Computing metrics for class: {label_name}")
            metrics[label_name] = {
                "accuracy": accuracy_score(y_true == encoded_label, y_pred == encoded_label),
                "precision": precision_score(y_true == encoded_label, y_pred == encoded_label),
                "recall": recall_score(y_true == encoded_label, y_pred == encoded_label),
                "f1_score": f1_score(y_true == encoded_label, y_pred == encoded_label),
            }
            self._plot_roc_curve(y_true == encoded_label, y_prob[:, encoded_label], label_name)
            self._plot_confusion_matrix(y_true == encoded_label, y_pred == encoded_label, label_name)

        self._plot_overall_confusion_matrix(y_true, y_pred)
        self._plot_overall_roc_curve(y_true, y_pred)

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

        roc_curve_dir = os.path.join(self.output_dir, "roc_curve")
        os.makedirs(roc_curve_dir, exist_ok=True)
        plt.savefig(os.path.join(roc_curve_dir, f"roc_curve_{label_name}.png"))
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

        confusion_matrix_dir = os.path.join(self.output_dir, "confusion_matrix")
        os.makedirs(confusion_matrix_dir, exist_ok=True)
        plt.savefig(
            os.path.join(confusion_matrix_dir, f"confusion_matrix_{label_name}.png")
        )
        plt.close()

    def _plot_overall_confusion_matrix(self, y_true, y_pred):
        """
        Plot the overall confusion matrix including all classes and save it to a file.
        """
        matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_mappings.keys(),
            yticklabels=self.label_mappings.keys(),
        )
        plt.title("Overall Confusion Matrix")
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, "overall_confusion_matrix.png"))
        plt.close()

    def _plot_overall_roc_curve(self, y_true, y_prob):
        """
        Plot the overall ROC curve including all classes and save it to a file.
        """
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_name, encoded_label in self.label_mappings.items():
            # Convert y_true and y_prob to binary arrays for the current class
            y_true_binary = (y_true == encoded_label).astype(int)
            y_prob_binary = (y_prob == encoded_label).astype(int)

            fpr[label_name], tpr[label_name], _ = roc_curve(
                y_true_binary, y_prob_binary
            )
            roc_auc[label_name] = auc(fpr[label_name], tpr[label_name])

        plt.figure(figsize=(10, 8))
        for label_name in self.label_mappings.keys():
            plt.plot(
                fpr[label_name],
                tpr[label_name],
                lw=2,
                label=f"{label_name} (AUC = {roc_auc[label_name]:.2f})",
            )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Overall ROC Curve")
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(self.output_dir, "overall_roc_curve.png"))
        plt.close()

    def _save_evaluation_metrics(self, metrics):
        """
        Save the evaluation metrics to a JSON file.
        """
        with open(os.path.join(self.output_dir, "eval_metrics.json"), "w") as file:
            json.dump(metrics, file, indent=4)
