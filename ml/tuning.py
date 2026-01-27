from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def tune_xgboost(X_train, y_train):
    param_grid = {
        "max_depth": [3, 5, 7],
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.05, 0.1]
    }

    model = xgb.XGBClassifier(
        scale_pos_weight=10,
        eval_metric="logloss",
        random_state=42
    )

    grid = GridSearchCV(
        model,
        param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_