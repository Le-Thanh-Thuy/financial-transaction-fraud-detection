import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_parquet(path)

def split_xy(df, target_col, drop_cols):
    X = df.drop(columns=[target_col] + drop_cols)
    y = (df[target_col] > 0).astype(int)
    return X, y

def train_test_split_xy(X, y, test_size, random_state):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )