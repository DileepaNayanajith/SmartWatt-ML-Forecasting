
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

    test_size = int(len(X) * 0.20)

    X_dev = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:]

    y_dev = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]

    return X_dev, X_test, y_dev, y_test
import time
import joblib
import numpy as np

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error

from src.models import get_models


PARAM_GRIDS = {
    "Ridge": {
        "model__alpha": [0.1, 1.0, 10.0]
    },
    "Decision Tree": {
        "model__max_depth": [5, 10],
        "model__min_samples_leaf": [2, 5]
    },
    "Random Forest": {
        "model__max_depth": [10, None],
        "model__min_samples_leaf": [2, 5]
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1]
    },
    "SVR": {
        "model__C": [1.0, 10.0],
        "model__epsilon": [0.01, 0.1]
    }
}


def train_models(X_dev, y_dev):
    models = get_models()
    cv = TimeSeriesSplit(n_splits=3)

    results = []
    predictions = []

    output_dir = Path("models/candidates")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, pipeline in models.items():
        print(f"\nTraining {name}...")

        best_mae = float("inf")
        best_params = None

        for params in ParameterGrid(PARAM_GRIDS[name]):
            fold_maes = []

            for fold, (train_idx, val_idx) in enumerate(
                cv.split(X_dev), start=1
            ):
                model = clone(pipeline)
                model.set_params(**params)

                X_train = X_dev.iloc[train_idx]
                y_train = y_dev.iloc[train_idx]
                X_val = X_dev.iloc[val_idx]
                y_val = y_dev.iloc[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                mae = mean_absolute_error(y_val, y_pred)
                fold_maes.append(mae)

            avg_mae = np.mean(fold_maes)

            if avg_mae < best_mae:
                best_mae = avg_mae
                best_params = params.copy()

        # Refit candidate on all development data
        best_model = clone(pipeline)
        best_model.set_params(**best_params)

        start = time.perf_counter()
        best_model.fit(X_dev, y_dev)
        training_time = time.perf_counter() - start

        filename = name.lower().replace(" ", "_")
        joblib.dump(
            best_model,
            output_dir / f"{filename}.joblib"
        )

        # Record out-of-fold validation predictions
        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_dev), start=1
        ):
            fold_model = clone(pipeline)
            fold_model.set_params(**best_params)

            fold_model.fit(
                X_dev.iloc[train_idx],
                y_dev.iloc[train_idx]
            )

            fold_pred = fold_model.predict(
                X_dev.iloc[val_idx]
            )

            for index, actual, predicted in zip(
                val_idx,
                y_dev.iloc[val_idx],
                fold_pred
            ):
                predictions.append({
                    "model": name,
                    "fold": fold,
                    "row_index": int(index),
                    "actual_kwh": float(actual),
                    "predicted_kwh": float(predicted)
                })

        results.append({
            "model": name,
            "cv_mae_kwh": float(best_mae),
            "best_params": str(best_params),
            "refit_seconds": training_time
        })

        print(
            f"{name}: CV MAE = {best_mae:.4f} kWh"
        )

    pd.DataFrame(results).to_csv(
        results_dir / "training_results.csv",
        index=False
    )

    pd.DataFrame(predictions).to_csv(
        results_dir / "validation_predictions.csv",
        index=False
    )

    return pd.DataFrame(results)



   
if __name__ == "__main__":
    X, y, df = load_training_data()

    X_dev, X_test, y_dev, y_test = split_data(X, y)

    print("Total observations:", len(df))
    print("Features:", X.shape[1])
    print("Development rows:", len(X_dev))
    print("Final test rows:", len(X_test))

    print("\nStarting ML training...")

    results = train_models(X_dev, y_dev)

    print("\nTraining completed!")
    print(results.to_string(index=False))
