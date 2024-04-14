import os

from ..data.data_loader import DataLoader
from ..preprocessing.preprocessor import PreprocessorBuilder
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

    def preprocess_data(self):
        data_loader = DataLoader(self.dataset_file_paths)
        logger.debug("DataLoader created.")

        all_data = data_loader.load_data(self.train_fraction)
        logger.info(f"Data loaded. Shape: {all_data.shape}")

        X = all_data.drop(columns=[" Label"])
        y = all_data[" Label"]
        logger.debug("Features (X) and labels (y) extracted from the loaded data.")

        preprocessor = (
            PreprocessorBuilder()
            .with_data_cleaning(fill_method="median")
            .with_correlated_feature_removal(
                correlation_threshold=self.correlation_threshold
            )
            .with_pca(pca_variance_ratio=self.pca_variance_ratio)
            .with_label_encoding()
            .with_one_hot_encoding()
            .build()
        )
        logger.debug(
            "Preprocessor built with data cleaning, correlated feature removal, PCA, and label encoding."
        )

        X_preprocessed, y_encoded, label_mappings = preprocessor.preprocess_data(
            X, y
        )
        # y_encoded = y_encoded_tuple[0]
        logger.info(
            f"Data preprocessing completed. Preprocessed features shape: {X_preprocessed.shape}"
        )

        return data_loader, X_preprocessed, y_encoded, label_mappings

    # def evaluate_model(
    #     self,
    #     model,
    #     pipeline_dir,
    #     X_train,
    #     y_train,
    #     X_val,
    #     y_val,
    #     X_test,
    #     y_test,
    #     label_mappings,
    #     history
    # ):
    #     evaluator = Evaluator(model, pipeline_dir, label_mappings)
    #     logger.debug("Evaluator created.")

    #     evaluator.evaluate(
    #         X_train,
    #         y_train,
    #         X_val,
    #         y_val,
    #         X_test,
    #         y_test,
    #         history
    #     )
    #     logger.info("Model evaluation completed.")
