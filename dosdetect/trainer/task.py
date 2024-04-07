import argparse
from datetime import datetime
import os

from .utils.logger import configure_logging

from .pipelines.bilstm_cnn_pipeline import BiLSTMCNNPipeline
from .pipelines.knn_pipeline import KNNPipeline
from .pipelines.random_forest_pipeline import RandomForestPipeline


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
        "--pipeline",
        type=str,
        default="bilstm-cnn",
        choices=["knn", "bilstm-cnn", "random-forest"],
        help='Choose the pipeline to run: "knn", "bilstm-cnn", or "random-forest" (default: "bilstm-cnn")',
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

    args = parser.parse_args()

    return args


def main():
    """
    Main function to run the selected pipeline.
    """
    args = parse_arguments()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.pipeline == "knn":
        pipeline_dir = os.path.join(args.model_dir, f"knn_pipeline_{timestamp}")
    elif args.pipeline == "bilstm-cnn":
        pipeline_dir = os.path.join(args.model_dir, f"bilstm_cnn_pipeline_{timestamp}")
    else:
        pipeline_dir = os.path.join(
            args.model_dir, f"random_forest_pipeline_{timestamp}"
        )

    os.makedirs(pipeline_dir, exist_ok=True)

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

    if args.pipeline == 'knn':
        pipeline_dir = os.path.join(args.model_dir, f"knn_pipeline_{timestamp}")
        pipeline = KNNPipeline(
            dataset_file_paths,
            pipeline_dir,
            args.auto_tune,
            args.knn_correlation_threshold,
            args.knn_pca_variance_ratio,
            args.knn_n_neighbors,
        )
    elif args.pipeline == "random-forest":
        pipeline_dir = os.path.join(
            args.model_dir, f"random_forest_pipeline_{timestamp}"
        )
        pipeline = RandomForestPipeline(
            dataset_file_paths,
            pipeline_dir,
            args.random_forest_correlation_threshold,
            args.random_forest_pca_variance_ratio,
            args.random_forest_n_estimators,
            args.random_forest_max_depth,
            args.random_forest_random_state,
        )
    elif args.pipeline == "bilstm-cnn":
        pipeline_dir = os.path.join(args.model_dir, f"bilstm_cnn_pipeline_{timestamp}")
        pipeline = BiLSTMCNNPipeline(
            dataset_file_paths,
            pipeline_dir,
            args.auto_tune,
            args.bilstm_cnn_correlation_threshold,
            args.bilstm_cnn_pca_variance_ratio,
            args.bilstm_cnn_epochs,
            args.bilstm_cnn_batch_size,
        )
        return pipeline
    else:
        raise NotImplementedError(
            "Not a valid pipeline. Run --help for a list of available pipelines."
        )

    pipeline.run()


if __name__ == '__main__':
    main()
