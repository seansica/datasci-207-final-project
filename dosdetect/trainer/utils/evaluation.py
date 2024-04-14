import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize
import numpy as np
import os
import json

from ..utils.logger import init_logger

logger = init_logger(__name__)


class BaseEvaluator:
    def __init__(self, model, output_dir, label_mappings):
        self.model = model
        self.label_mappings = label_mappings
        self.output_dir = os.path.expanduser(output_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        logger.debug("Evaluator initialized with the trained model.")

    def _save_evaluation_metrics(self, metrics):
        with open(os.path.join(self.output_dir, "eval_metrics.json"), "w") as file:
            json.dump(metrics, file, indent=4)


class KerasEvaluator(BaseEvaluator):
    def evaluate(self, X_test, y_test, history):
        logger.debug("Evaluating Keras model.")
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        self._plot_training_curves(history)
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        metrics = self._compute_metrics_keras(y_test, y_pred, y_pred_prob)
        self._save_evaluation_metrics(metrics)
        logger.debug("Model evaluation completed.")
        return metrics

    def _compute_metrics_keras(self, y_true, y_pred, y_prob):
        metrics = {}
        y_true_labels = np.argmax(y_true, axis=1)
        for label_name, encoded_label in self.label_mappings.items():
            logger.info(f"Computing metrics for class: {label_name}")
            y_true_binary = y_true_labels == encoded_label
            y_pred_binary = y_pred == encoded_label
            metrics[label_name] = {
                "accuracy": accuracy_score(y_true_binary, y_pred_binary),
                "precision": precision_score(y_true_binary, y_pred_binary),
                "recall": recall_score(y_true_binary, y_pred_binary),
                "f1_score": f1_score(y_true_binary, y_pred_binary),
            }
            self._plot_roc_curve(y_true_binary, y_prob[:, encoded_label], label_name)
            self._plot_confusion_matrix(y_true_binary, y_pred_binary, label_name)

        self._plot_overall_confusion_matrix(y_true_labels, y_pred)
        self._plot_overall_roc_curve(y_true_labels, y_prob)

        return metrics

    def _plot_training_curves(self, history):
        self._plot_curves(
            history.history["loss"],
            history.history["val_loss"],
            "Training and Validation Loss",
            "loss",
        )
        self._plot_curves(
            history.history["accuracy"],
            history.history["val_accuracy"],
            "Training and Validation Accuracy",
            "accuracy",
        )

    def _plot_curves(self, train_curve, val_curve, title, filename):
        plt.figure()
        plt.plot(train_curve, label="Training")
        plt.plot(val_curve, label="Validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(filename.capitalize())
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{filename}_curve.png"))
        plt.close()

    def _plot_roc_curve(self, y_true, y_prob, label_name):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {label_name}")
        plt.legend(loc="lower right")

        roc_curve_dir = os.path.join(self.output_dir, "roc_curve")
        os.makedirs(roc_curve_dir, exist_ok=True)
        plt.savefig(os.path.join(roc_curve_dir, f"roc_curve_{label_name}.png"))
        plt.close()

    def _plot_confusion_matrix(self, y_true, y_pred, label_name):
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
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_name, encoded_label in self.label_mappings.items():
            y_true_binary = (y_true == encoded_label).astype(int)

            if len(y_prob.shape) == 2:
                y_prob_binary = y_prob[:, encoded_label]
            else:
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


class SKLearnEvaluator(BaseEvaluator):
    def evaluate(self, X_test, y_test):
        logger.debug("Evaluating scikit-learn model.")
        y_pred = self.model.predict(X_test)
        y_pred_prob = self.model.predict_proba(X_test)

        # metrics = self._compute_metrics_sklearn(y_test, y_pred, y_pred_prob)
        if len(y_test.shape) == 1 and len(y_pred.shape) == 1:
            metrics = self._compute_metrics_sklearn_1d(y_test, y_pred, y_pred_prob)
        else:
            metrics = self._compute_metrics_sklearn(y_test, y_pred, y_pred_prob)

        self._save_evaluation_metrics(metrics)
        logger.debug("Model evaluation completed.")
        return metrics

    def _compute_metrics_sklearn(self, y_true, y_pred, y_prob):
        metrics = {}

        # Iterate over each class using its index and label name
        for i, label_name in enumerate(self.label_mappings.keys()):
            logger.info(f"Computing metrics for class: {label_name}")

            # Isolate the outcomes and predictions for the current class
            y_true_binary = y_true[:, i]
            y_pred_binary = y_pred[:, i]

            # Adjusting based on the shape of y_prob[i]
            if y_prob[i].shape[1] == 1:
                # If only one probability score is present, use it directly
                class_probabilities = y_prob[i].flatten()
            else:
                # Otherwise, use the second column as the probability of the positive class
                class_probabilities = y_prob[i][:, 1]

            # Calculate various metrics for the current class
            metrics[label_name] = {
                "accuracy": accuracy_score(y_true_binary, y_pred_binary),
                "precision": precision_score(
                    y_true_binary, y_pred_binary, average="binary", zero_division=0
                ),
                "recall": recall_score(
                    y_true_binary, y_pred_binary, average="binary", zero_division=0
                ),
                "f1_score": f1_score(
                    y_true_binary, y_pred_binary, average="binary", zero_division=0
                ),
                "auc": roc_auc_score(y_true_binary, class_probabilities),
            }

            # Plot the ROC curve for the current class
            self._plot_roc_curve_sklearn(y_true_binary, class_probabilities, label_name)
            self._plot_confusion_matrix_for_class(
                y_true_binary, y_pred_binary, label_name
            )

        self._plot_combined_roc_curve(y_true, y_prob)
        self._plot_global_confusion_matrix(y_true, y_pred)

        return metrics

    def _plot_roc_curve_sklearn(self, y_true_binary, class_probabilities, label_name):
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true_binary, class_probabilities)
        roc_auc = auc(fpr, tpr)

        # Plotting the ROC curve
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {label_name}")
        plt.legend(loc="lower right")

        # Define the directory and save the plot
        roc_curve_dir = os.path.join(self.output_dir, "roc_curve_sklearn")
        os.makedirs(roc_curve_dir, exist_ok=True)
        plt.savefig(os.path.join(roc_curve_dir, f"roc_curve_{label_name}.png"))
        plt.close()

    def _plot_confusion_matrix_for_class(
        self, y_true_binary, y_pred_binary, label_name
    ):
        """
        Plot and save the confusion matrix for a single class.

        Parameters:
        - y_true_binary: 1D array of binary true labels for the class.
        - y_pred_binary: 1D array of binary predictions for the class.
        - label_name: Name of the class.
        """
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not " + label_name, label_name],
            yticklabels=["Not " + label_name, label_name],
        )
        plt.title(f"Confusion Matrix - {label_name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()

        confusion_matrix_dir = os.path.join(self.output_dir, "confusion_matrices")
        os.makedirs(confusion_matrix_dir, exist_ok=True)
        plt.savefig(
            os.path.join(confusion_matrix_dir, f"confusion_matrix_{label_name}.png")
        )
        plt.close()

    def _plot_global_confusion_matrix(self, y_true, y_pred):
        """
        Plot and save a global confusion matrix that encompasses all classes.

        Parameters:
        - y_true: 2D array of shape (n_samples, n_classes), true labels.
        - y_pred: 2D array of shape (n_samples, n_classes), predictions.
        """
        # Convert the 2D binary arrays into single vectors of class labels
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true_labels, y_pred_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_mappings.keys(),
            yticklabels=self.label_mappings.keys(),
        )
        plt.title("Global Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()

    def _plot_combined_roc_curve(self, y_true, y_prob):
        """
        Plot a combined ROC curve for all classes on the same graph.

        Parameters:
        - y_true: 2D array of shape (n_samples, n_classes), true labels in a one-hot encoded format.
        - y_prob: 3D array where each element is the 2D array of shape (n_samples, n_classes) representing the predicted probabilities for each class.
        """
        plt.figure(figsize=(10, 8))

        # Process each class
        for i, label_name in enumerate(self.label_mappings.keys()):
            # Assuming y_true is one-hot encoded, thus need to select the correct column for each class
            y_true_binary = y_true[:, i]

            # Adjusting based on the shape of y_prob[i]
            if y_prob[i].shape[1] == 1:
                # If only one probability score is present, use it directly
                class_probabilities = y_prob[i].flatten()
            else:
                # Otherwise, use the second column as the probability of the positive class
                class_probabilities = y_prob[i][:, 1]

            fpr, tpr, _ = roc_curve(y_true_binary, class_probabilities)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label=f"{label_name} (area = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Combined ROC Curve for All Classes")
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(self.output_dir, "roc_curves.png"))
        plt.close()

    def _compute_metrics_sklearn_1d(self, y_true, y_pred, y_prob):
        metrics = {}

        # Iterate over each class using its label name
        for label_name in self.label_mappings.keys():
            logger.info(f"Computing metrics for class: {label_name}")

            # Convert labels to binary format for the current class
            y_true_binary = (y_true == self.label_mappings[label_name]).astype(int)
            y_pred_binary = (y_pred == self.label_mappings[label_name]).astype(int)

            # Get the probabilities for the current class
            class_probabilities = y_prob[:, self.label_mappings[label_name]]

            # Calculate various metrics for the current class
            metrics[label_name] = {
                "accuracy": accuracy_score(y_true_binary, y_pred_binary),
                "precision": precision_score(
                    y_true_binary, y_pred_binary, average="binary", zero_division=0
                ),
                "recall": recall_score(
                    y_true_binary, y_pred_binary, average="binary", zero_division=0
                ),
                "f1_score": f1_score(
                    y_true_binary, y_pred_binary, average="binary", zero_division=0
                ),
                "auc": roc_auc_score(y_true_binary, class_probabilities),
            }

            # Plot the ROC curve for the current class
            self._plot_roc_curve_sklearn(y_true_binary, class_probabilities, label_name)
            self._plot_confusion_matrix_for_class(
                y_true_binary, y_pred_binary, label_name
            )

        self._plot_combined_roc_curve_1d(y_true, y_prob)
        self._plot_global_confusion_matrix_1d(y_true, y_pred)

        return metrics

    def _plot_combined_roc_curve_1d(self, y_true, y_prob):
        """
        Plot a combined ROC curve for all classes on the same graph.

        Parameters:
        - y_true: 1D array of shape (n_samples,), true labels.
        - y_prob: 2D array of shape (n_samples, n_classes), predicted probabilities for each class.
        """
        plt.figure(figsize=(10, 8))

        # Process each class
        for label_name, label_index in self.label_mappings.items():
            # Convert labels to binary format for the current class
            y_true_binary = (y_true == label_index).astype(int)

            # Get the probabilities for the current class
            class_probabilities = y_prob[:, label_index]

            fpr, tpr, _ = roc_curve(y_true_binary, class_probabilities)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label=f"{label_name} (area = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Combined ROC Curve for All Classes")
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(self.output_dir, "roc_curves.png"))
        plt.close()

    def _plot_global_confusion_matrix_1d(self, y_true, y_pred):
        """
        Plot and save a global confusion matrix that encompasses all classes.

        Parameters:
        - y_true: 1D array of shape (n_samples,), true labels.
        - y_pred: 1D array of shape (n_samples,), predicted labels.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(self.label_mappings.keys()),
            yticklabels=list(self.label_mappings.keys()),
        )
        plt.title("Global Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()
