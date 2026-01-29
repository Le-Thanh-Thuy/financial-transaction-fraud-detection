# ml/config.py
DATA_PATH = "spark-data/feature/transaction_features"
MODEL_PATH = "spark-data/models/final_fraud_model"
METRIC_PATH = "spark-data/models/metrics.csv"
TARGET_COL = "isFraud"

TRAIN_SIZE = 0.8
RANDOM_SEED = 42