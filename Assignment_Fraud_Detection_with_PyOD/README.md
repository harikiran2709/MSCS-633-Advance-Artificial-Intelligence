## Overview
This project implements fraud detection on credit card transactions using PyOD's AutoEncoder algorithm for anomaly detection.


## Dataset
**Kaggle Credit Card Fraud Detection Dataset**: https://www.kaggle.com/datasets/whenamancodes/fraud-detection

The script automatically uses the real Kaggle dataset if `creditcard.csv` is present in the project directory.

## Requirements
- Python 3.8+
- PyOD, pandas, numpy, scikit-learn, matplotlib, tensorflow

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python fraud_detection.py
```

## Implementation
- **Deep Learning**: PyOD AutoEncoder neural network
- **Anomaly Detection**: Reconstruction error-based fraud detection
- **Performance**: AUC-ROC score of 0.9584 on real dataset
- **Dataset**: 284,807 credit card transactions

## Output
- Console metrics (AUC-ROC, classification report, confusion matrix)
- Visualization dashboard saved as `fraud_detection_results.png`