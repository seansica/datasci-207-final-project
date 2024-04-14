import os

from ..data.data_loader import DataLoader
from ..preprocessing.data_cleaner import DataCleaner
from ..preprocessing.preprocessor import Preprocessor
from ..utils.logger import init_logger

logger = init_logger(__name__)


class BasePipeline:

    def __init__(
        self,
        dataset_file_paths,
        pipeline_dir,
        auto_tune,
        train_fraction,
        correlation_threshold,
        pca_variance_ratio,
    ):
        self.dataset_file_paths = dataset_file_paths
        self.pipeline_dir = os.path.expanduser(pipeline_dir)
        self.auto_tune = auto_tune
        self.train_fraction = train_fraction
        self.correlation_threshold = correlation_threshold
        self.pca_variance_ratio = pca_variance_ratio

        logger.debug(
            f"BasePipeline initialized with file paths: {dataset_file_paths}, "
            f"pipeline directory: {self.pipeline_dir}, "
            f"hyperparameter auto-tune: {auto_tune}, "
            f"training fraction: {train_fraction}, "
            f"correlation threshold: {correlation_threshold}, "
            f"PCA variance ratio: {pca_variance_ratio}"
        )

    def initialize(self, preprocessor: Preprocessor):
        
        # Load the data
        data_loader = DataLoader(self.dataset_file_paths)
        logger.debug("DataLoader created.")
        all_data = data_loader.load_data(self.train_fraction)
        logger.info(f"Data loaded. Shape: {all_data.shape}")

        # Sanitize column names immediately after loading
        data_cleaner = DataCleaner()
        all_data = data_cleaner.sanitize_column_names(all_data)

        # Split X and y (labels)
        X = all_data.drop(columns=["label"])
        y = all_data["label"]
        logger.debug("Features (X) and labels (y) extracted from the loaded data.")

        logger.info(
            "Preprocessor built with data cleaning, correlated feature removal, PCA, and label encoding."
        )

        # Apply feature engineering
        X_preprocessed, y_encoded, label_mappings = preprocessor.transform(X, y)

        logger.info(
            f"Data preprocessing completed. Preprocessed features shape: {X_preprocessed.shape}"
        )

        return data_loader, X_preprocessed, y_encoded, label_mappings
