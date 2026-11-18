
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit

DATA_PATH = Path(
    "part1_output/data/processed/hourly_features.csv"
)
GROUPS_PATH = Path(
    "part1_output/data/processed/feature_groups.json"
)


def load_training_data():
    df = pd.read_csv(DATA_PATH)

    df["target_hour"] = pd.to_datetime(
        df["target_hour"]
    )

    df = df.sort_values("target_hour").reset_index(drop=True)

    with open(GROUPS_PATH, "r") as f:
        groups = json.load(f)

    # Use the 40 primary features from Part 1
    feature_cols = [
        feature
        for group in ["A", "B", "C", "D"]
        for feature in groups[group]
    ]

    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[feature_cols]
    y = df["y_kwh"]

    return X, y, df


def split_data(X, y):
    # Reserve the last 20% for final testing
    test_size = int(len(X) * 0.20)

    X_dev = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:]

    y_dev = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]

    return X_dev, X_test, y_dev, y_test


if __name__ == "__main__":
    X, y, df = load_training_data()

    X_dev, X_test, y_dev, y_test = split_data(X, y)

    print("Total observations:", len(df))
    print("Number of features:", X.shape[1])
    print("Development rows:", len(X_dev))
    print("Final test rows:", len(X_test))

    cv = TimeSeriesSplit(n_splits=3)

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_dev), start=1
    ):
        print(
            f"Fold {fold}: "
            f"Train={len(train_idx)}, "
            f"Validation={len(val_idx)}"
        )

