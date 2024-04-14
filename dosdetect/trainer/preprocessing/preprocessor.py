from enum import Enum

from sklearn.preprocessing import LabelEncoder

from .encoder import LabelEncoderHandler
from .correlation import CorrelatedFeatureRemover
from .data_cleaner import DataCleaner
from .pca import PCATransformer

from ..utils.logger import init_logger

logger = init_logger(__name__)


class PreprocessorOptions(Enum):
    DATA_CLEANING = 0
    COLUMN_SANITIZATION = 1
    CORRELATED_FEATURE_REMOVAL = 2
    PCA = 3
    DATA_SCALING = 4
    SPARSE_ENCODING = 5
    ONE_HOT_ENCODING = 6


class PreprocessorBuilder:
    """
    Builder class for constructing a Preprocessor instance with specified preprocessing steps.
    """

    def __init__(self):
        """
        Initialize the PreprocessorBuilder with an empty list of preprocessing steps.
        """
        self.steps = []
        logger.debug("PreprocessorBuilder initialized with empty steps.")

    def with_one_hot_encoding(self):
        """
        Add one-hot encoding step to the preprocessing pipeline.

        Returns:
            PreprocessorBuilder: The builder instance with the one-hot encoding step added.
        """
        self.steps.append(
            (PreprocessorOptions.ONE_HOT_ENCODING, LabelEncoderHandler(mode="one_hot"))
        )
        logger.debug("One-hot encoding step added to PreprocessorBuilder.")
        return self

    def with_sparse_encoding(self):
        """
        Add sparse label encoding step to the preprocessing pipeline.

        Returns:
            PreprocessorBuilder: The builder instance with the sparse encoding step added.
        """
        self.steps.append(
            (PreprocessorOptions.SPARSE_ENCODING, LabelEncoderHandler(mode="sparse"))
        )
        logger.debug("Sparse encoding step added to PreprocessorBuilder.")
        return self

    def with_correlated_feature_removal(self, correlation_threshold=0.9):
        """
        Add correlated feature removal step to the preprocessing pipeline.

        Args:
            correlation_threshold (float, optional): The threshold for determining highly correlated features.
                Defaults to 0.9.

        Returns:
            PreprocessorBuilder: The builder instance with the correlated feature removal step added.
        """
        self.steps.append(
            (
                PreprocessorOptions.CORRELATED_FEATURE_REMOVAL,
                CorrelatedFeatureRemover(correlation_threshold),
            )
        )
        logger.debug(f"Correlated feature removal step added to PreprocessorBuilder with threshold {correlation_threshold}.")
        return self

    def with_pca(self, pca_variance_ratio=0.95):
        """
        Add PCA transformation step to the preprocessing pipeline.

        Args:
            pca_variance_ratio (float, optional): The desired amount of variance to retain in the PCA transformation.
                Defaults to 0.95.

        Returns:
            PreprocessorBuilder: The builder instance with the PCA transformation step added.
        """
        self.steps.append((PreprocessorOptions.PCA, PCATransformer(pca_variance_ratio)))
        logger.debug(f"PCA transformation step added to PreprocessorBuilder with variance ratio {pca_variance_ratio}.")
        return self

    def with_data_cleaning(self, fill_method='median'):
        """
        Add data cleaning step to the preprocessing pipeline.

        Args:
            fill_method (str, optional): The method for filling infinite and NaN values.
                Supported values: 'median', 'mean', or any other value. Defaults to 'median'.

        Returns:
            PreprocessorBuilder: The builder instance with the data cleaning step added.
        """
        self.steps.append((PreprocessorOptions.DATA_CLEANING, DataCleaner(fill_method)))
        logger.debug(f"Data cleaning step added to PreprocessorBuilder with fill method '{fill_method}'.")
        return self

    def with_column_sanitization(self):
        self.steps.append(PreprocessorOptions.COLUMN_SANITIZATION, DataCleaner())
        logger.debug(f"Column sanitization step added to PreprocessorBuilder.")
        return self

    def with_data_scaling(self, scaler):
        self.steps.append((PreprocessorOptions.DATA_SCALING, scaler))
        logger.debug(f"Data scaling step added to PreprocessorBuilder.")
        return self

    def build(self):
        """
        Build and return the Preprocessor instance with the specified preprocessing steps.

        Returns:
            Preprocessor: The constructed Preprocessor instance.
        """
        preprocessor = Preprocessor(self.steps)
        logger.debug("Preprocessor instance built from PreprocessorBuilder.")
        return preprocessor


class Preprocessor:
    """
    Preprocessor class for applying a sequence of preprocessing steps to the data.
    """

    def __init__(self, steps):
        """
        Initialize the Preprocessor with the specified preprocessing steps.

        Args:
            steps (list): A list of preprocessing steps to be applied in sequence.
        """
        self.steps = steps
        logger.debug(f"Preprocessor initialized with {len(steps)} preprocessing steps.")

    def transform(self, X, y=None):
        """
        Preprocess the input data by applying the specified preprocessing steps.

        Args:
            X (pandas.DataFrame): The input features.
            y (pandas.Series, optional): The target labels. Defaults to None.
            num_classes (int, optional): The number of unique classes. Required for one-hot encoding.

        Returns:
            tuple: A tuple containing the preprocessed features (X) labels (y), and label_mappings.
                Label mappings is not returned if the LABEL_ENCODING step is not enabled.
        """
        logger.info("Starting data preprocessing...")

        label_mappings = None  # to be set by encoding steps

        for step, transformer in self.steps:
            if step == PreprocessorOptions.DATA_CLEANING:
                logger.info("Cleaning data...")
                X = transformer.clean_data(X)

            elif step == PreprocessorOptions.COLUMN_SANITIZATION:
                logger.info("Sanitizing column names...")
                X = transformer.sanitize_column_names(X)

            elif step == PreprocessorOptions.CORRELATED_FEATURE_REMOVAL:
                logger.info("Removing correlated features...")
                X = transformer.remove_correlated_features(X)

            elif step == PreprocessorOptions.PCA:
                logger.info("Applying PCA transformation...")
                X = transformer.apply_pca(X)

            elif step == PreprocessorOptions.DATA_SCALING:
                logger.info("Applying data scaling...")
                X = transformer.fit_transform(X)

            elif step == PreprocessorOptions.SPARSE_ENCODING:
                logger.info("Applying label encoding...")
                y_encoded, label_mappings = transformer.encode_labels(y)

            elif step == PreprocessorOptions.ONE_HOT_ENCODING:
                logger.info("Applying one-hot encoding...")
                y_encoded, label_mappings = transformer.encode_labels(y)

        logger.info("Data preprocessing completed.")

        return (X, y_encoded, label_mappings) if label_mappings else X
