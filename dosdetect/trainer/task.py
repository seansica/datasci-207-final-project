import argparse
import logging

from .config import Config
from .utils.logger import setup_logger

from .pipelines.bilstm_cnn_pipeline import BiLSTMCNNPipeline
from .pipelines.knn_pipeline import KNNPipeline


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run KNN or BiLSTM-CNN pipeline.')
    parser.add_argument('--pipeline', type=str, default='bilstm-cnn', choices=['knn', 'bilstm-cnn'],
                        help='Choose the pipeline to run: "knn" or "bilstm-cnn" (default: "bilstm-cnn")')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to store log files (default: "~/.dosdetect/logs")')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save trained models (default: "~/.dosdetect/models")')
    args = parser.parse_args()

    Config.set_log_dir(args.log_dir)
    Config.set_model_dir(args.model_dir)

    return args


def main():
    """
    Main function to run the selected pipeline.
    """
    args = parse_arguments()

    # Note that parse_arguments must be called before setting up the logger because the log directory is set in the args
    logger = setup_logger('task_logger', 'task.log', level=logging.DEBUG)

    logger.info(f"Starting {args.pipeline} pipeline...")

    root_path = '/Users/seansica/OneDrive - Sica/Education/Berkeley/W207-Applied-ML/datasci-207-final-project/datasets/CICIDS2017/MachineLearningCVE'

    file_paths = [
        f'{root_path}/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        f'{root_path}/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        f'{root_path}/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        f'{root_path}/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        f'{root_path}/Friday-WorkingHours-Morning.pcap_ISCX.csv',
        f'{root_path}/Tuesday-WorkingHours.pcap_ISCX.csv',
        f'{root_path}/Monday-WorkingHours.pcap_ISCX.csv',
        f'{root_path}/Wednesday-workingHours.pcap_ISCX.csv'
    ]

    if args.pipeline == 'knn':
        pipeline = KNNPipeline(file_paths)
    else:
        pipeline = BiLSTMCNNPipeline(file_paths)

    pipeline.run()
    logger.info(f"{args.pipeline} pipeline finished.")


if __name__ == '__main__':
    main()