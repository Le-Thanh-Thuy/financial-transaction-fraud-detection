import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

def chi2_feature_selection(X, y, k=20):
    """
    Chi-square feature selection (phù hợp classification & fraud)
    """
    selector = SelectKBest(score_func=chi2, k=min(k, X.shape[1]))
    selector.fit(X, y)

    selected_cols = X.columns[selector.get_support()]
    return X[selected_cols], selected_cols

def basic_feature_stats(X):
    """
    Dùng cho EDA nhanh
    """
    return pd.DataFrame({
        "mean": X.mean(),
        "std": X.std(),
        "missing_rate": X.isna().mean()
    })