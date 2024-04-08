# pipelines/logistic_regression_pipeline.py
import json
import os
from sklearn.preprocessing import StandardScaler

from .base_pipeline import BasePipeline
from ..utils.evaluation import SKLearnEvaluator
from ..data.data_loader import DataLoader
from ..preprocessing.preprocessor import PreprocessorBuilder
from ..models.logistic_regression import LogisticRegressionModel
from ..utils.logger import init_logger

logger = init_logger("logistic_regression_pipeline_logger")


class LogisticRegressionPipeline(BasePipeline):

    def __init__(
        self,
        dataset_file_paths,
        pipeline_dir,
        auto_tune=True,
        train_fraction=1.0,
        correlation_threshold=None,
        pca_variance_ratio=None,
        C=1.0,
        max_iter=100,
        random_state=None,
    ):
        super().__init__(
            dataset_file_paths,
            pipeline_dir,
            auto_tune,
            train_fraction,
            correlation_threshold,
            pca_variance_ratio,
        )
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def run(self):
        logger.info("Starting Logistic Regression pipeline...")

        pipeline_details = {
            "pipeline_type": "LogisticRegression",
            "auto_tune": self.auto_tune,
            "train_fraction": self.train_fraction,
            "correlation_threshold": self.correlation_threshold,
            "pca_variance_ratio": self.pca_variance_ratio,
            "C": self.C,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }

        with open(os.path.join(self.pipeline_dir, "pipeline_details.json"), "w") as f:
            json.dump(pipeline_details, f)

        # data_loader, X_preprocessed, y_encoded, label_mappings = self.preprocess_data()

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
            .with_data_scaling(scaler=StandardScaler())
            .with_label_encoding()
            # .with_one_hot_encoding()
            .build()
        )
        logger.debug(
            "Preprocessor built with data cleaning, correlated feature removal, PCA, and label encoding."
        )

        X_preprocessed, y_encoded, label_mappings = preprocessor.preprocess_data(X, y)
        # y_encoded = y_encoded_tuple[0]
        logger.info(
            f"Data preprocessing completed. Preprocessed features shape: {X_preprocessed.shape}"
        )

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data(
            X_preprocessed, y_encoded
        )

        logger.debug(
            f"Data split into train, validation, and test sets. "
            f"Train: {X_train.shape}, {y_train.shape}, "
            f"Validation: {X_val.shape}, {y_val.shape}, "
            f"Test: {X_test.shape}, {y_test.shape}"
        )

        logistic_regression = LogisticRegressionModel(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            auto_tune=self.auto_tune,
        )
        logistic_regression.build_model()
        logger.info("Logistic Regression model initialized.")

        logistic_regression.train(X_train, y_train)
        logger.info("Logistic Regression model trained.")

        logistic_regression.save_model(self.pipeline_dir)
        logger.info("Logistic Regression model saved.")

        evaluator = SKLearnEvaluator(
            logistic_regression.model, self.pipeline_dir, label_mappings
        )
        evaluator.evaluate(X_test, y_test)

        logger.info("Logistic Regression pipeline finished.")
