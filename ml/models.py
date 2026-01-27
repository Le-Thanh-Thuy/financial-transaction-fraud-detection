import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from config import XGB_PARAMS

def build_xgboost():
    return xgb.XGBClassifier(**XGB_PARAMS)

def build_logistic():
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

def get_model(name="xgb"):
    if name == "xgb":
        return build_xgboost()
    elif name == "logistic":
        return build_logistic()
    else:
        raise ValueError(f"Unknown model: {name}")