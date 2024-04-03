import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
import re
import logging
from ..utils.logger import setup_logger


logger = setup_logger('preprocessor_logger', 'preprocessor.log', level=logging.DEBUG)


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
    
    def with_label_encoding(self):
        """
        Add label encoding step to the preprocessing pipeline.

        Returns:
            PreprocessorBuilder: The builder instance with the label encoding step added.
        """
        self.steps.append(LabelEncoder())
        logger.debug("Label encoding step added to PreprocessorBuilder.")
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
        self.steps.append(CorrelatedFeatureRemover(correlation_threshold))
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
        self.steps.append(PCATransformer(pca_variance_ratio))
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
        self.steps.append(DataCleaner(fill_method))
        logger.debug(f"Data cleaning step added to PreprocessorBuilder with fill method '{fill_method}'.")
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
    
    def preprocess_data(self, X, y=None):
        """
        Preprocess the input data by applying the specified preprocessing steps.

        Args:
            X (pandas.DataFrame): The input features.
            y (pandas.Series, optional): The target labels. Defaults to None.

        Returns:
            tuple: A tuple containing the preprocessed features (X) and labels (y).
        """
        logger.info("Starting data preprocessing...")
        X = self.sanitize_column_names(X)
        
        for step in self.steps:
            if isinstance(step, LabelEncoder):
                logger.info("Applying label encoding...")
                y = self.encode_labels(y, step)
            elif isinstance(step, CorrelatedFeatureRemover):
                logger.info("Removing correlated features...")
                X = step.remove_correlated_features(X)
            elif isinstance(step, PCATransformer):
                logger.info("Applying PCA transformation...")
                X = step.apply_pca(X)
            elif isinstance(step, DataCleaner):
                logger.info("Cleaning data...")
                X = step.clean_data(X)
        
        logger.info("Data preprocessing completed.")
        return X, y
    
    def encode_labels(self, y, label_encoder):
        """
        Encode the target labels using the specified label encoder.

        Args:
            y (pandas.Series): The target labels.
            label_encoder (LabelEncoder): The label encoder instance.

        Returns:
            numpy.ndarray: The encoded labels in categorical format.
        """
        logger.debug("Encoding labels...")
        y_encoded = label_encoder.fit_transform(y)
        
        # Print the mapping of original labels to encoded values
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        logger.debug(f"Label Mapping: {label_mapping}")
        
        num_classes = len(label_encoder.classes_)
        y_categorical = to_categorical(y_encoded, num_classes=num_classes)
        
        logger.debug("Label encoding completed.")
        return y_categorical, label_mapping

    def sanitize_column_names(self, X):
        """
        Sanitize the column names of the input features.

        Args:
            X (pandas.DataFrame): The input features.

        Returns:
            pandas.DataFrame: The input features with sanitized column names.
        """
        logger.debug("Sanitizing column names...")
        
        # Modify column names in place:
        #   - Convert column names to lowercase
        #   - Replace non-alphanumeric characters with underscores
        #   - Strip leading and trailing whitespaces
        X.columns = X.columns.str.strip().str.replace(' ', '_').str.replace('-', '_').str.lower()
        
        logger.debug("Column names sanitized.")
        
        # Return the modified DataFrame
        return X

class CorrelatedFeatureRemover:
    """
    Class for removing highly correlated features from the dataset.
    """

    def __init__(self, correlation_threshold=0.9):
        """
        Initialize the CorrelatedFeatureRemover with the specified correlation threshold.

        Args:
            correlation_threshold (float, optional): The threshold for determining highly correlated features.
                Defaults to 0.9.
        """
        self.correlation_threshold = correlation_threshold
        logger.debug(f"CorrelatedFeatureRemover initialized with threshold {correlation_threshold}.")
    
    def remove_correlated_features(self, X):
        """
        Remove highly correlated features from the dataset.

        Args:
            X (pandas.DataFrame): The input features.

        Returns:
            pandas.DataFrame: The dataset with selected features after removing highly correlated ones.
        """
        logger.debug("Removing correlated features...")
        # Compute the correlation matrix
        correlation_matrix = X.corr()
        correlated_features = set()
        
        # Iterate over the upper triangle of the correlation matrix
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > self.correlation_threshold:
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)
        
        # Drop the correlated features from the dataset
        selected_features = X.columns.drop(correlated_features)
        X_selected = X[selected_features]
        
        logger.debug(f"Removed {len(correlated_features)} correlated features.")
        return X_selected


class PCATransformer:
    """
    Class for applying PCA transformation to the dataset.
    """

    def __init__(self, pca_variance_ratio=0.95):
        """
        Initialize the PCATransformer with the specified variance ratio.

        Args:
            pca_variance_ratio (float, optional): The desired amount of variance to retain in the PCA transformation.
                Defaults to 0.95.
        """
        self.pca_variance_ratio = pca_variance_ratio
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_variance_ratio)
        logger.debug(f"PCATransformer initialized with variance ratio {pca_variance_ratio}.")
    
    def apply_pca(self, X):
        """
        Apply PCA transformation to the dataset.

        Args:
            X (pandas.DataFrame): The input features.

        Returns:
            numpy.ndarray: The transformed dataset after applying PCA.
        """
        logger.debug("Applying PCA transformation...")
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA transformation
        X_pca = self.pca.fit_transform(X_scaled)
        
        logger.debug(f"PCA transformation applied. Transformed dataset shape: {X_pca.shape}")
        return X_pca


class DataCleaner:
    """
    Class for cleaning the dataset by handling infinite and NaN values.
    """

    def __init__(self, fill_method='median'):
        """
        Initialize the DataCleaner with the specified fill method.

        Args:
            fill_method (str, optional): The method for filling infinite and NaN values.
                Supported values: 'median', 'mean', or any other value. Defaults to 'median'.
        """
        self.fill_method = fill_method
        logger.debug(f"DataCleaner initialized with fill method '{fill_method}'.")
    
    def clean_data(self, data):
        """
        Clean the dataset by handling infinite and NaN values.

        Args:
            data (pandas.DataFrame): The input dataset.

        Returns:
            pandas.DataFrame: The cleaned dataset with infinite and NaN values handled.
        """
        logger.debug("Cleaning data...")

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, but got {type(data)} instead.")

        # Replace infinite values with NaN using pandas methods
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Check and fill NaN values based on the selected fill method
        for column in data.columns:
            if data[column].isnull().any():  # Check if there are any NaN values in the column
                if self.fill_method == 'median':
                    fill_value = data[column].median()
                elif self.fill_method == 'mean':
                    fill_value = data[column].mean()
                elif self.fill_method == 'zero':
                    fill_value = 0
                else:
                    # Assuming self.fill_method is a callable that returns a scalar
                    # This path needs specific handling based on what self.fill_method is
                    fill_value = self.fill_method(data[column])

                # Assign the result of fillna directly to the DataFrame column
                data[column] = data[column].fillna(fill_value)

        logger.debug("Data cleaning completed.")
        return data