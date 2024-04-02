# pipelines/knn_pipeline.py
from trainer.data.data_loader import DataLoader
from trainer.preprocessing.preprocessor import PreprocessorBuilder
from trainer.models.knn import KNN
from trainer.utils.evaluation import Evaluator

import logging
from trainer.utils.logger import setup_logger

logger = setup_logger('knn_pipeline_logger', 'knn_pipeline.log', level=logging.DEBUG)


class KNNPipeline:
    """
    Pipeline for training and evaluating a KNN model.
    """

    def __init__(self, file_paths, correlation_threshold=0.9, pca_variance_ratio=0.95, n_neighbors=5):
        """
        Initialize the KNNPipeline.

        Args:
            file_paths (list): List of file paths containing the data.
            correlation_threshold (float): Threshold for removing correlated features.
            pca_variance_ratio (float): Ratio of variance to retain in PCA.
            n_neighbors (int): Number of neighbors to consider in KNN.
        """
        self.file_paths = file_paths
        self.correlation_threshold = correlation_threshold
        self.pca_variance_ratio = pca_variance_ratio
        self.n_neighbors = n_neighbors
        logger.debug(f"KNNPipeline initialized with file paths: {file_paths}, "
                     f"correlation threshold: {correlation_threshold}, "
                     f"PCA variance ratio: {pca_variance_ratio}, "
                     f"n_neighbors: {n_neighbors}")

    def run(self):
        """
        Run the KNN pipeline.
        """
        logger.info("Starting KNN pipeline...")

        data_loader = DataLoader(self.file_paths)
        logger.debug("DataLoader created.")

        all_data = data_loader.load_data()
        logger.info(f"Data loaded. Shape: {all_data.shape}")

        X = all_data.drop(columns=['Label'])
        y = all_data['label']
        logger.debug("Features (X) and labels (y) extracted from the loaded data.")

        preprocessor = PreprocessorBuilder() \
            .with_data_cleaning(fill_method='median') \
            .with_correlated_feature_removal(correlation_threshold=self.correlation_threshold) \
            .with_pca(pca_variance_ratio=self.pca_variance_ratio) \
            .with_label_encoding() \
            .build()
        logger.debug("Preprocessor built with data cleaning, correlated feature removal, PCA, and label encoding.")

        X_preprocessed, y_encoded = preprocessor.preprocess_data(X, y)
        logger.info(f"Data preprocessing completed. Preprocessed features shape: {X_preprocessed.shape}")

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data(X_preprocessed, y_encoded)
        logger.debug(f"Data split into train, validation, and test sets. "
                     f"Train: {X_train.shape}, {y_train.shape}, "
                     f"Validation: {X_val.shape}, {y_val.shape}, "
                     f"Test: {X_test.shape}, {y_test.shape}")

        knn = KNN(n_neighbors=self.n_neighbors)
        logger.debug("KNN model initialized.")

        knn.train(X_train, y_train)
        logger.info("KNN model trained.")

        evaluator = Evaluator(knn)
        logger.debug("Evaluator created.")

        evaluator.evaluate(X_test, y_test)
        logger.info("Model evaluation completed.")

        logger.info("KNN pipeline finished.")