import pandas as pd
from sklearn.model_selection import train_test_split

import logging
from ..utils.logger import init_logger

logger = init_logger("data_loader_logger")


class DataLoader:
    """
    A class for loading and splitting data from CSV files.
    """

    def __init__(self, file_paths):
        """
        Initialize the DataLoader with file paths.

        Args:
            file_paths (list): List of file paths to CSV files.
        """
        self.file_paths = file_paths
        logger.debug(f"DataLoader initialized with file paths: {file_paths}")

    def load_data(self, fraction=1.0):
        """
        Load a fraction of data from CSV files while maintaining class distribution.

        Args:
            fraction (float): Fraction of data to load (default: 1.0).

        Returns:
            pandas.DataFrame: Loaded data as a DataFrame.
        """
        logger.info(f"Loading {fraction*100}% of data from CSV files...")
        data_frames = []

        # Iterate over each file path
        for file_path in self.file_paths:
            # Read CSV file into a DataFrame
            df = pd.read_csv(file_path)
            logger.debug(f"Loaded data from {file_path}. Shape: {df.shape}")

            # Get the unique labels and their counts
            label_counts = df[" Label"].value_counts()
            labels = label_counts.index
            counts = label_counts.values

            # Calculate the number of samples to load for each label
            num_samples = (counts * fraction).astype(int)

            # Iterate over each label and load the corresponding samples
            loaded_data = []
            for label, count, num in zip(labels, counts, num_samples):
                if num > 0:
                    loaded_data.append(df[df[" Label"] == label][:num])

            # Concatenate the loaded data for the current file
            loaded_df = pd.concat(loaded_data, ignore_index=True)
            data_frames.append(loaded_df)
            logger.debug(
                f"Loaded {fraction*100}% of data from {file_path}. Shape: {loaded_df.shape}"
            )

        # Concatenate all loaded DataFrames into a single DataFrame
        all_data = pd.concat(data_frames, ignore_index=True)
        logger.info(
            f"{fraction*100}% of data loaded and concatenated. Shape: {all_data.shape}"
        )

        return all_data

    def split_data(self, X, y, train_size=0.6, val_size=0.2, test_size=0.2):
        """
        Split the data into train, validation, and test sets.

        Args:
            X (pandas.DataFrame): Input features.
            y (pandas.Series): Target labels.
            train_size (float): Proportion of data to include in the train set.
            val_size (float): Proportion of data to include in the validation set.
            test_size (float): Proportion of data to include in the test set.

        Returns:
            tuple: A tuple containing the split data:
                - (X_train, y_train): Train set features and labels.
                - (X_val, y_val): Validation set features and labels.
                - (X_test, y_test): Test set features and labels.
        """
        logger.info("Splitting data into train, validation, and test sets...")

        # Calculate the ratios for train and validation sets
        train_ratio = train_size / (train_size + val_size + test_size)
        val_ratio = val_size / (train_size + val_size + test_size)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_ratio, random_state=42)
        logger.debug(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

        # Split the test set further into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_size/(test_size+val_size), random_state=42)
        logger.debug(f"Validation set size: {len(X_val)}, Test set size: {len(X_test)}")

        logger.info("Data split completed.")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
