import argparse
from datetime import datetime
import os

from .utils.logger import configure_logging

from .pipelines.bilstm_cnn_pipeline import BiLSTMCNNPipeline
from .pipelines.knn_pipeline import KNNPipeline
from .pipelines.random_forest_pipeline import RandomForestPipeline
from .pipelines.logistic_regression_pipeline import LogisticRegressionPipeline
from .pipelines.gradient_boosted_trees_pipeline import GradientBoostedTreesPipeline
from .pipelines.gru_pipeline import GRUPipeline
from .pipelines.decision_tree_pipeline import DecisionTreePipeline
from .pipelines.ffnn_pipeline import FFNNPipeline

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run KNN, BiLSTM-CNN, or Random Forest pipeline."
    )
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Use KerasTuner for automatic hyperparameter tuning",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/Users/seansica/OneDrive - Sica/Education/Berkeley/W207-Applied-ML/datasci-207-final-project/datasets/CICIDS2017/MachineLearningCVE",
        help="Choose the path to your dataset/training files.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Train the model on a fraction of the training dataset to speed up training.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="bilstm-cnn",
        choices=[
            "bilstm-cnn",
            "knn",
            "random-forest",
            "logistic-regression",
            "gradient-boosted-trees",
            "ffnn",
        ],
        help='Choose the pipeline to run: "knn", "bilstm-cnn", "random-forest", "logistic-regression", "gradient-boosted-trees", or "ffnn" (default: "bilstm-cnn")',
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="~/.dosdetect",
        help='Directory to store log files (default: "~/.dosdetect/logs")',
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="~/.dosdetect",
        help='Directory to save trained models (default: "~/.dosdetect/models")',
    )

    # KNN pipeline hyperparameters
    parser.add_argument(
        "--knn-correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for KNN pipeline (default: 0.9)",
    )
    parser.add_argument(
        "--knn-pca-variance-ratio",
        type=float,
        default=0.95,
        help="PCA variance ratio for KNN pipeline (default: 0.95)",
    )
    parser.add_argument(
        "--knn-n-neighbors",
        type=int,
        default=5,
        help="Number of neighbors for KNN pipeline (default: 5)",
    )

    # BiLSTM-CNN pipeline hyperparameters
    parser.add_argument(
        "--bilstm-cnn-correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for BiLSTM-CNN pipeline (default: 0.9)",
    )
    parser.add_argument(
        "--bilstm-cnn-pca-variance-ratio",
        type=float,
        default=0.95,
        help="PCA variance ratio for BiLSTM-CNN pipeline (default: 0.95)",
    )
    parser.add_argument(
        "--bilstm-cnn-epochs",
        type=int,
        default=10,
        help="Number of epochs for BiLSTM-CNN pipeline (default: 10)",
    )
    parser.add_argument(
        "--bilstm-cnn-batch-size",
        type=int,
        default=32,
        help="Batch size for BiLSTM-CNN pipeline (default: 32)",
    )

    # Random Forest pipeline hyperparameters
    parser.add_argument(
        "--random-forest-correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for Random Forest pipeline (default: 0.9)",
    )
    parser.add_argument(
        "--random-forest-pca-variance-ratio",
        type=float,
        default=0.95,
        help="PCA variance ratio for Random Forest pipeline (default: 0.95)",
    )
    parser.add_argument(
        "--random-forest-n-estimators",
        type=int,
        default=100,
        help="Number of estimators for Random Forest pipeline (default: 100)",
    )
    parser.add_argument(
        "--random-forest-max-depth",
        type=int,
        default=None,
        help="Maximum depth for Random Forest pipeline (default: None)",
    )
    parser.add_argument(
        "--random-forest-random-state",
        type=int,
        default=None,
        help="Random state for Random Forest pipeline (default: None)",
    )

    # Logistic Regression pipeline hyperparameters
    parser.add_argument(
        "--logistic-regression-correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for Logistic Regression pipeline (default: 0.9)",
    )
    parser.add_argument(
        "--logistic-regression-pca-variance-ratio",
        type=float,
        default=0.95,
        help="PCA variance ratio for Logistic Regression pipeline (default: 0.95)",
    )
    parser.add_argument(
        "--logistic-regression-C",
        type=float,
        default=1.0,
        help="Inverse of regularization strength for Logistic Regression pipeline (default: 1.0)",
    )
    parser.add_argument(
        "--logistic-regression-max-iter",
        type=int,
        default=100,
        help="Maximum number of iterations for Logistic Regression pipeline (default: 100)",
    )
    parser.add_argument(
        "--logistic-regression-random-state",
        type=int,
        default=None,
        help="Random state for Logistic Regression pipeline (default: None)",
    )

    # Gradient Boosted Trees pipeline hyperparameters
    parser.add_argument(
        "--gradient-boosted-trees-correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for Gradient Boosted Trees pipeline (default: 0.9)",
    )
    parser.add_argument(
        "--gradient-boosted-trees-pca-variance-ratio",
        type=float,
        default=0.95,
        help="PCA variance ratio for Gradient Boosted Trees pipeline (default: 0.95)",
    )
    parser.add_argument(
        "--gradient-boosted-trees-max-depth",
        type=int,
        default=3,
        help="Maximum depth for Gradient Boosted Trees pipeline (default: 3)",
    )
    parser.add_argument(
        "--gradient-boosted-trees-learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for Gradient Boosted Trees pipeline (default: 0.1)",
    )
    parser.add_argument(
        "--gradient-boosted-trees-n-estimators",
        type=int,
        default=100,
        help="Number of estimators for Gradient Boosted Trees pipeline (default: 100)",
    )

    # GRU pipeline hyperparameters
    parser.add_argument(
        "--gru-correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for GRU pipeline (default: 0.9)",
    )
    parser.add_argument(
        "--gru-pca-variance-ratio",
        type=float,
        default=0.95,
        help="PCA variance ratio for GRU pipeline (default: 0.95)",
    )
    parser.add_argument(
        "--gru-epochs",
        type=int,
        default=10,
        help="Number of epochs for GRU pipeline (default: 10)",
    )
    parser.add_argument(
        "--gru-batch-size",
        type=int,
        default=32,
        help="Batch size for GRU pipeline (default: 32)",
    )

    # Decision Tree pipeline hyperparameters
    parser.add_argument(
        "--decision-tree-correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for Decision Tree pipeline (default: 0.9)",
    )
    parser.add_argument(
        "--decision-tree-pca-variance-ratio",
        type=float,
        default=0.95,
        help="PCA variance ratio for Decision Tree pipeline (default: 0.95)",
    )
    parser.add_argument(
        "--decision-tree-max-depth",
        type=int,
        default=None,
        help="Maximum depth for Decision Tree pipeline (default: None)",
    )
    parser.add_argument(
        "--decision-tree-min-samples-split",
        type=int,
        default=2,
        help="Minimum number of samples required to split an internal node for Decision Tree pipeline (default: 2)",
    )
    parser.add_argument(
        "--decision-tree-min-samples-leaf",
        type=int,
        default=1,
        help="Minimum number of samples required to be at a leaf node for Decision Tree pipeline (default: 1)",
    )
    parser.add_argument(
        "--decision-tree-criterion",
        type=str,
        default="gini",
        choices=["gini", "entropy"],
        help="The function to measure the quality of a split for Decision Tree pipeline (default: 'gini')",
    )

    # FFNN pipeline hyperparameters
    parser.add_argument(
        "--ffnn-correlation-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for FFNN pipeline (default: 0.9)",
    )
    parser.add_argument(
        "--ffnn-pca-variance-ratio",
        type=float,
        default=0.95,
        help="PCA variance ratio for FFNN pipeline (default: 0.95)",
    )
    parser.add_argument(
        "--ffnn-hidden-units",
        type=int,
        default=128,
        help="Number of hidden units for FFNN pipeline (default: 128)",
    )
    parser.add_argument(
        "--ffnn-dropout-rate",
        type=float,
        default=0.2,
        help="Dropout rate for FFNN pipeline (default: 0.2)",
    )
    parser.add_argument(
        "--ffnn-num-hidden-layers",
        type=int,
        default=2,
        help="Number of hidden layers for FFNN pipeline (default: 2)",
    )
    parser.add_argument(
        "--ffnn-epochs",
        type=int,
        default=10,
        help="Number of epochs for FFNN pipeline (default: 10)",
    )
    parser.add_argument(
        "--ffnn-batch-size",
        type=int,
        default=32,
        help="Batch size for FFNN pipeline (default: 32)",
    )

    args = parser.parse_args()

    return args


def init_pipeline_dir(model_dir, pipeline_name):
    """Initialize the pipeline directory.

    This function creates a new directory for the pipeline with a unique timestamp appended to the name.
    If the directory already exists, it will not be recreated.

    Args:
        model_dir (str): The base directory where the pipeline directory will be created.
        pipeline_name (str): The name of the pipeline.

    Returns:
        str: The path to the newly created pipeline directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = os.path.join(model_dir, f"{pipeline_name}_{timestamp}")
    os.makedirs(pipeline_dir, exist_ok=True)
    return pipeline_dir


def main():
    """
    Main function to run the selected pipeline.
    """
    args = parse_arguments()

    # Initialize a directory to store all logs and files generated files during the pipeline execution
    pipeline_dir = init_pipeline_dir(args.model_dir, f"{args.pipeline}_pipeline")

    # Configure the root logger
    configure_logging(pipeline_dir)

    dataset_file_paths = [
        f"{args.dataset}/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        f"{args.dataset}/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        f"{args.dataset}/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        f"{args.dataset}/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        f"{args.dataset}/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        f"{args.dataset}/Tuesday-WorkingHours.pcap_ISCX.csv",
        f"{args.dataset}/Monday-WorkingHours.pcap_ISCX.csv",
        f"{args.dataset}/Wednesday-workingHours.pcap_ISCX.csv",
    ]

    if args.pipeline == "knn":
        pipeline = KNNPipeline(
            dataset_file_paths,
            pipeline_dir,
            args.auto_tune,
            args.train_fraction,
            args.knn_correlation_threshold,
            args.knn_pca_variance_ratio,
            args.knn_n_neighbors,
        )
    elif args.pipeline == "random-forest":
        pipeline = RandomForestPipeline(
            dataset_file_paths,
            pipeline_dir,
            args.auto_tune,
            args.train_fraction,
            args.random_forest_correlation_threshold,
            args.random_forest_pca_variance_ratio,
            args.random_forest_n_estimators,
            args.random_forest_max_depth,
            args.random_forest_random_state,
        )
    elif args.pipeline == "logistic-regression":
        pipeline = LogisticRegressionPipeline(
            dataset_file_paths,
            pipeline_dir,
            args.auto_tune,
            args.train_fraction,
            args.logistic_regression_correlation_threshold,
            args.logistic_regression_pca_variance_ratio,
            args.logistic_regression_C,
            args.logistic_regression_max_iter,
            args.logistic_regression_random_state,
        )
    elif args.pipeline == "bilstm-cnn":
        pipeline = BiLSTMCNNPipeline(
            dataset_file_paths,
            pipeline_dir,
            args.auto_tune,
            args.train_fraction,
            args.bilstm_cnn_correlation_threshold,
            args.bilstm_cnn_pca_variance_ratio,
            args.bilstm_cnn_epochs,
            args.bilstm_cnn_batch_size,
        )
    elif args.pipeline == "gradient-boosted-trees":
        pipeline = GradientBoostedTreesPipeline(
            dataset_file_paths,
            pipeline_dir,
            args.auto_tune,
            args.train_fraction,
            args.gradient_boosted_trees_correlation_threshold,
            args.gradient_boosted_trees_pca_variance_ratio,
            args.gradient_boosted_trees_max_depth,
            args.gradient_boosted_trees_learning_rate,
            args.gradient_boosted_trees_n_estimators,
        )
    elif args.pipeline == "gru":
        pipeline = GRUPipeline(
            dataset_file_paths,
            pipeline_dir,
            args.auto_tune,
            args.train_fraction,
            args.gru_correlation_threshold,
            args.gru_pca_variance_ratio,
            args.gru_epochs,
            args.gru_batch_size,
        )
    elif args.pipeline == "decision-tree":
        pipeline = DecisionTreePipeline(
            dataset_file_paths,
            pipeline_dir,
            args.auto_tune,
            args.train_fraction,
            args.decision_tree_correlation_threshold,
            args.decision_tree_pca_variance_ratio,
            args.decision_tree_max_depth,
            args.decision_tree_min_samples_split,
            args.decision_tree_min_samples_leaf,
            args.decision_tree_criterion,
        )
    elif args.pipeline == "ffnn":
        pipeline = FFNNPipeline(
            dataset_file_paths,
            pipeline_dir,
            args.auto_tune,
            args.train_fraction,
            args.ffnn_correlation_threshold,
            args.ffnn_pca_variance_ratio,
            args.ffnn_hidden_units,
            args.ffnn_dropout_rate,
            args.ffnn_num_hidden_layers,
            args.ffnn_epochs,
            args.ffnn_batch_size,
        )
    else:
        raise NotImplementedError(
            "Not a valid pipeline. Run --help for a list of available pipelines."
        )

    pipeline.run()


if __name__ == '__main__':
    main()
