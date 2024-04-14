from ..utils.logger import init_logger


logger = init_logger(__name__)


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
        logger.debug(
            f"CorrelatedFeatureRemover initialized with threshold {correlation_threshold}."
        )

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
