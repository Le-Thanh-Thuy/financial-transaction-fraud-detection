# ml/config.py

DATA_PATH = "spark-data/gold/fraud_features"

TARGET_COL = "fraud_count"
DROP_COLS = ["user_id"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# XGBoost default params
XGB_PARAMS = {
    "max_depth": 5,
    "n_estimators": 100,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 10,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE
}
