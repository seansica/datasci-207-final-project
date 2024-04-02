import argparse
from trainer.pipelines.bilstm_cnn_pipeline import BiLSTMCNNPipeline
from trainer.pipelines.knn_pipeline import KNNPipeline

import logging
from trainer.utils.logger import setup_logger

logger = setup_logger('task_logger', 'task.log', level=logging.DEBUG)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run KNN or BiLSTM-CNN pipeline.')
    parser.add_argument('--pipeline', type=str, default='bilstm-cnn', choices=['knn', 'bilstm-cnn'],
                        help='Choose the pipeline to run: "knn" or "bilstm-cnn" (default: "bilstm-cnn")')
    args = parser.parse_args()
    return args


def main():
    """
    Main function to run the selected pipeline.
    """
    args = parse_arguments()
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