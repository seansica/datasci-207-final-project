# DosDetect

The 'dosdetect' package can be installed with pip and interacted with using a command-line interface (CLI). The CLI provides 23 configurable flags to customize the pipeline according to your needs.

The package supports several machine learning models, including K-Nearest Neighbors (KNN), Random Forest (RF), Logistic Regression (LR), and a hybrid model of Bidirectional Long Short-Term Memory and Convolutional Neural Network (Bi-LSTM-CNN).

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