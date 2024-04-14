import numpy as np
import pandas as pd
from ..utils.logger import init_logger

logger = init_logger(__name__)


class DataCleaner:
    """
    Class for cleaning the dataset by handling infinite and NaN values.
    """

    def __init__(self, fill_method="median"):
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
            raise TypeError(
                f"Expected a pandas DataFrame, but got {type(data)} instead."
            )

        # Replace infinite values with NaN using pandas methods
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Check and fill NaN values based on the selected fill method
        for column in data.columns:
            if (
                data[column].isnull().any()
            ):  # Check if there are any NaN values in the column
                if self.fill_method == "median":
                    fill_value = data[column].median()
                elif self.fill_method == "mean":
                    fill_value = data[column].mean()
                elif self.fill_method == "zero":
                    fill_value = 0
                else:
                    # Assuming self.fill_method is a callable that returns a scalar
                    # This path needs specific handling based on what self.fill_method is
                    fill_value = self.fill_method(data[column])

                # Assign the result of fillna directly to the DataFrame column
                data[column] = data[column].fillna(fill_value)

        logger.debug("Data cleaning completed.")
        return data

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
        X.columns = (
            X.columns.str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
            .str.lower()
        )

        logger.debug("Column names sanitized.")

        # Return the modified DataFrame
        return X
