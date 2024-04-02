from trainer.data.data_loader import DataLoader
from trainer.preprocessing.preprocessor import PreprocessorBuilder
from trainer.models.bilstm_cnn import BiLSTMCNN
from trainer.utils.evaluation import Evaluator

import logging
from ..utils.logger import setup_logger

logger = setup_logger('bilstm_cnn_pipeline_logger', 'bilstm_cnn_pipeline.log', level=logging.DEBUG)


class BiLSTMCNNPipeline:
    """
    Pipeline for training and evaluating a BiLSTM-CNN model.
    """

    def __init__(self, file_paths, correlation_threshold=0.9, pca_variance_ratio=0.95):
        """
        Initialize the BiLSTMCNNPipeline.

        Args:
            file_paths (list): List of file paths containing the data.
            correlation_threshold (float): Threshold for removing correlated features.
            pca_variance_ratio (float): Ratio of variance to retain in PCA.
        """
        self.file_paths = file_paths
        self.correlation_threshold = correlation_threshold
        self.pca_variance_ratio = pca_variance_ratio
        logger.debug(f"BiLSTMCNNPipeline initialized with file paths: {file_paths}, "
                     f"correlation threshold: {correlation_threshold}, "
                     f"PCA variance ratio: {pca_variance_ratio}")

    def run(self):
        """
        Run the BiLSTM-CNN pipeline.
        """
        logger.info("Starting BiLSTM-CNN pipeline...")

        # Create a DataLoader instance
        data_loader = DataLoader(self.file_paths)
        logger.debug("DataLoader created.")

        # Load the data
        all_data = data_loader.load_data()
        logger.info(f"Data loaded. Shape: {all_data.shape}")

        # Extract features (X) and labels (y) from the loaded data
        X = all_data.drop(columns=[' Label'])
        y = all_data[' Label']
        logger.debug("Features (X) and labels (y) extracted from the loaded data.")

        # Build the preprocessor with specified preprocessing steps
        preprocessor = PreprocessorBuilder() \
            .with_data_cleaning(fill_method='median') \
            .with_correlated_feature_removal(correlation_threshold=0.9) \
            .with_pca(pca_variance_ratio=0.95) \
            .with_label_encoding() \
            .build()
        logger.debug("Preprocessor built with data cleaning, correlated feature removal, PCA, and label encoding.")

        # Preprocess the data
        X_preprocessed, y_encoded = preprocessor.preprocess_data(X, y)
        logger.info(f"Data preprocessing completed. Preprocessed features shape: {X_preprocessed.shape}")

        # Get the number of classes from the encoded labels
        num_classes = y_encoded.shape[1]
        logger.debug(f"Number of classes: {num_classes}")

        # Split the data into train, validation, and test sets
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data(X_preprocessed, y_encoded)
        logger.debug(f"Data split into train, validation, and test sets. "
                     f"Train: {X_train.shape}, {y_train.shape}, "
                     f"Validation: {X_val.shape}, {y_val.shape}, "
                     f"Test: {X_test.shape}, {y_test.shape}")

        # Create the BiLSTM-CNN model
        input_shape = (X_train.shape[1], 1)
        bilstm_cnn = BiLSTMCNN(input_shape, num_classes)
        model = bilstm_cnn.create_model()
        logger.info("BiLSTM-CNN model created.")

        # Set the number of training epochs
        epochs = 10

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.debug("Model compiled with Adam optimizer, categorical cross-entropy loss, and accuracy metric.")

        # Train the model
        logger.info(f"Training the model for {epochs} epochs...")
        model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))
        logger.info("Model training completed.")

        # Create an Evaluator instance
        evaluator = Evaluator(model)
        logger.debug("Evaluator created.")

        # Evaluate the model on the test set
        evaluator.evaluate(X_test, y_test)
        logger.info("Model evaluation completed.")

        logger.info("BiLSTM-CNN pipeline finished.")