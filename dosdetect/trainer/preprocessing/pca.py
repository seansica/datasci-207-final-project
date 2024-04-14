from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ..utils.logger import init_logger


logger = init_logger(__name__)


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
        logger.debug(
            f"PCATransformer initialized with variance ratio {pca_variance_ratio}."
        )

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

        logger.debug(
            f"PCA transformation applied. Transformed dataset shape: {X_pca.shape}"
        )
        return X_pca
