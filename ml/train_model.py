from config import *
from utils import load_data, split_xy, train_test_split_xy
from features import chi2_feature_selection
from models import get_model
from metrics import evaluate_model
# from tuning import tune_xgboost

def main():
    # Load
    df = load_data(DATA_PATH)

    # Split X, y
    X, y = split_xy(df, TARGET_COL, DROP_COLS)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split_xy(
        X, y, TEST_SIZE, RANDOM_STATE
    )

    # Feature selection
    X_train_fs, selected_cols = chi2_feature_selection(X_train, y_train, k=20)
    X_test_fs = X_test[selected_cols]

    # Model
    model = get_model("xgb")

    # Train
    model.fit(X_train_fs, y_train)

    # Evaluate
    evaluate_model(model, X_test_fs, y_test)

if __name__ == "__main__":
    main()