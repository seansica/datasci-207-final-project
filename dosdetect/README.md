# DosDetect

The 'dosdetect' package can be installed with pip and interacted with using a command-line interface (CLI). The CLI provides 47 configurable flags to customize the pipeline according to your needs.

The package supports several machine learning models, including K-Nearest Neighbors (KNN), Random Forest (RF), Logistic Regression (LR), Bidirectional Long Short-Term Memory and Convolutional Neural Network (Bi-LSTM-CNN), Gradient Boosted Decision Trees, Gated Recurrent Unit (GRU), Decision Tree, and Feed-Forward Neural Network (FFNN).

## Installation

You can install the 'dosdetect' package using pip directly from the source code:

```bash
pip install .
```

Alternatively, you can build a wheel distribution of the package and then install it:

```bash
python setup.py bdist_wheel
pip install dist/*.whl
```

## Usage

After installation, you can execute the 'dosdetect' tool using the `dosdetect` command in your terminal. 

To see a list of supported arguments, you can pass the `--help` flag:

```bash
dosdetect --help
```

Key flags include:

- **`--dataset`**: This flag allows users to specify the path to the CIC-IDS2017 dataset, making the application adaptable to other sequential CSV datasets with a 'Label' classification feature.
- **`--train-fraction`**: Users can train the model on a subset of the original dataset, which proves invaluable during initial development and debugging phases.
- **`--log-dir`** and **`--model-dir`**: These flags are used to designate directories for storing logs, exported files (such as model parameters in .keras and .pkl formats), evaluated performances, and model details. By default, all pipelines create a unique directory within `~/.dosdetect` appended with a timestamp suffix.

The CLI further supports individual model configurations through additional flags specific to each model type:

| Argument                                      | Default Value   | Description                                                                                |
| --------------------------------------------- | --------------- | ------------------------------------------------------------------------------------------ |
| `--auto-tune`                                 | False           | Use KerasTuner for automatic hyperparameter tuning                                         |
| `--dataset`                                   | Path to dataset | Choose the path to your dataset/training files                                             |
| `--train-fraction`                            | 1.0             | Train the model on a fraction of the training dataset to speed up training                 |
| `--pipeline`                                  | "bilstm-cnn"    | Choose the pipeline to run: "knn", "bilstm-cnn", "random-forest", "logistic-regression", "gradient-boosted-trees", "gru", "decision-tree", or "ffnn" |
| `--log-dir`                                   | "~/.dosdetect"  | Directory to store log files                                                               |
| `--model-dir`                                 | "~/.dosdetect"  | Directory to save trained models                                                           |
| `--knn-correlation-threshold`                 | 0.9             | Correlation threshold for KNN pipeline                                                     |
| `--knn-pca-variance-ratio`                    | 0.95            | PCA variance ratio for KNN pipeline                                                        |
| `--knn-n-neighbors`                           | 5               | Number of neighbors for KNN pipeline                                                       |
| `--bilstm-cnn-correlation-threshold`          | 0.9             | Correlation threshold for BiLSTM-CNN pipeline                                              |
| `--bilstm-cnn-pca-variance-ratio`             | 0.95            | PCA variance ratio for BiLSTM-CNN pipeline                                                 |
| `--bilstm-cnn-epochs`                         | 10              | Number of epochs for BiLSTM-CNN pipeline                                                   |
| `--bilstm-cnn-batch-size`                     | 32              | Batch size for BiLSTM-CNN pipeline                                                         |
| `--random-forest-correlation-threshold`       | 0.9             | Correlation threshold for Random Forest pipeline                                           |
| `--random-forest-pca-variance-ratio`          | 0.95            | PCA variance ratio for Random Forest pipeline                                              |
| `--random-forest-n-estimators`                | 100             | Number of estimators for Random Forest pipeline                                            |
| `--random-forest-max-depth`                   | None            | Maximum depth for Random Forest pipeline                                                   |
| `--random-forest-random-state`                | None            | Random state for Random Forest pipeline                                                    |
| `--logistic-regression-correlation-threshold` | 0.9             | Correlation threshold for Logistic Regression pipeline                                     |
| `--logistic-regression-pca-variance-ratio`    | 0.95            | PCA variance ratio for Logistic Regression pipeline                                        |
| `--logistic-regression-C`                     | 1.0             | Inverse of regularization strength for Logistic Regression pipeline                        |
| `--logistic-regression-max-iter`              | 100             | Maximum number of iterations for Logistic Regression pipeline                              |
| `--logistic-regression-random-state`          | None            | Random state for Logistic Regression pipeline                                              |
| `--gradient-boosted-trees-correlation-threshold` | 0.9          | Correlation threshold for Gradient Boosted Trees pipeline                                  |
| `--gradient-boosted-trees-pca-variance-ratio`| 0.95            | PCA variance ratio for Gradient Boosted Trees pipeline                                     |
| `--gradient-boosted-trees-max-depth`         | 3               | Maximum depth for Gradient Boosted Trees pipeline                                          |
| `--gradient-boosted-trees-learning-rate`     | 0.1             | Learning rate for Gradient Boosted Trees pipeline                                          |
| `--gradient-boosted-trees-n-estimators`      | 100             | Number of estimators for Gradient Boosted Trees pipeline                                   |
| `--gru-correlation-threshold`                | 0.9             | Correlation threshold for GRU pipeline                                                     |
| `--gru-pca-variance-ratio`                   | 0.95            | PCA variance ratio for GRU pipeline                                                        |
| `--gru-epochs`                               | 10              | Number of epochs for GRU pipeline                                                          |
| `--gru-batch-size`                           | 32              | Batch size for GRU pipeline                                                                |
| `--decision-tree-correlation-threshold`      | 0.9             | Correlation threshold for Decision Tree pipeline                                            |
| `--decision-tree-pca-variance-ratio`         | 0.95            | PCA variance ratio for Decision Tree pipeline                                              |
| `--decision-tree-max-depth`                  | None            | Maximum depth for Decision Tree pipeline                                                   |
| `--decision-tree-min-samples-split`          | 2               | Minimum number of samples required to split an internal node for Decision Tree pipeline    |
| `--decision-tree-min-samples-leaf`           | 1               | Minimum number of samples required to be at a leaf node for Decision Tree pipeline         |
| `--decision-tree-criterion`                  | "gini"          | The function to measure the quality of a split for Decision Tree pipeline                  |
| `--ffnn-correlation-threshold`               | 0.9             | Correlation threshold for FFNN pipeline                                                    |
| `--ffnn-pca-variance-ratio`                  | 0.95            | PCA variance ratio for FFNN pipeline                                                       |
| `--ffnn-hidden-units`                        | 128             | Number of hidden units for FFNN pipeline                                                   |
| `--ffnn-dropout-rate`                        | 0.2             | Dropout rate for FFNN pipeline                                                             |
| `--ffnn-num-hidden-layers`                   | 2               | Number of hidden layers for FFNN pipeline                                                  |
| `--ffnn-epochs`                              | 10              | Number of epochs for FFNN pipeline                                                          |
| `--ffnn-batch-size`                          | 32              | Batch size for FFNN pipeline                                                                |
