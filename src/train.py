
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.models import get_models
from src.baselines import predict_baselines


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

DATA_PATH = Path(
    "part1_output/data/processed/hourly_features.csv"
)

GROUPS_PATH = Path(
    "part1_output/data/processed/feature_groups.json"
)

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models/candidates")

RANDOM_STATE = 42
N_SPLITS = 3

FEATURE_GROUPS = ["A", "B", "C", "D"]


# ==========================================
# 2. HYPERPARAMETER GRIDS
# ==========================================

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


# ==========================================
# 3. LOAD PART 1 DATASET
# ==========================================

def load_training_data():

    print("\nLoading SmartWatt dataset...")

    df = pd.read_csv(DATA_PATH)

    df["target_hour"] = pd.to_datetime(
        df["target_hour"]
    )

    # Chronological order is essential
    df = df.sort_values(
        "target_hour"
    ).reset_index(drop=True)

    if df["target_hour"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    if not df["target_hour"].diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "Dataset must contain consecutive hourly records."
        )

    with open(GROUPS_PATH, "r") as file:
        groups = json.load(file)

    feature_cols = [
        feature
        for group in FEATURE_GROUPS
        for feature in groups[group]
    ]

    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("Duplicate feature names found.")

    missing = set(feature_cols) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing features: {sorted(missing)}"
        )

    if "y_kwh" not in df.columns:
        raise ValueError("Target column y_kwh missing.")

    X = df[feature_cols].copy()
    y = df["y_kwh"].copy()

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Missing values found.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature values found.")

    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite target values found.")

    if (y <= 0).any():
        raise ValueError("Target values must be positive.")

    print("Dataset shape:", df.shape)
    print("Number of features:", X.shape[1])

    return X, y, df


# ==========================================
# 4. CHRONOLOGICAL DATA SPLIT
# ==========================================

def split_data(X, y):

    # Keep last 20% untouched for Part 3
    test_size = int(len(X) * 0.20)

    X_dev = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:]

    y_dev = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]

    print("\nChronological data split:")
    print("Development rows:", len(X_dev))
    print("Final test rows:", len(X_test))

    return X_dev, X_test, y_dev, y_test


# ==========================================
# 5. EVALUATION METRICS
# ==========================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return {
        "mae_kwh": float(mae),
        "rmse_kwh": float(rmse),
        "r2": float(r2)
    }


# ==========================================
# 6. EVALUATE THREE BASELINES
# ==========================================

def evaluate_baselines(df, X_dev, y_dev):

    print("\n================================")
    print("EVALUATING BASELINE FORECASTS")
    print("================================")

    cv = TimeSeriesSplit(n_splits=N_SPLITS)

    dev_df = df.iloc[:len(X_dev)]

    baseline_predictions = predict_baselines(
        dev_df
    )

    results = []
    prediction_records = []

    for name in baseline_predictions.columns:

        print(f"\nBaseline: {name}")

        fold_metrics = []

        for fold, (_, val_idx) in enumerate(
            cv.split(X_dev), start=1
        ):

            actual = y_dev.iloc[val_idx]

            predicted = baseline_predictions[
                name
            ].iloc[val_idx]

            metrics = calculate_metrics(
                actual, predicted
            )

            fold_metrics.append(metrics)

            print(
                f"Fold {fold}: "
                f"MAE={metrics['mae_kwh']:.4f}, "
                f"RMSE={metrics['rmse_kwh']:.4f}, "
                f"R2={metrics['r2']:.4f}"
            )

            for index, true_value, pred_value in zip(
                val_idx, actual, predicted
            ):
                prediction_records.append({
                    "model": name,
                    "fold": fold,
                    "row_index": int(index),
                    "target_hour": str(
                        dev_df.iloc[index]["target_hour"]
                    ),
                    "actual_kwh": float(true_value),
                    "predicted_kwh": float(pred_value)
                })

        results.append({
            "model": name,
            "cv_mae_kwh": np.mean([
                m["mae_kwh"] for m in fold_metrics
            ]),
            "cv_rmse_kwh": np.mean([
                m["rmse_kwh"] for m in fold_metrics
            ]),
            "cv_r2": np.mean([
                m["r2"] for m in fold_metrics
            ])
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_DIR / "baseline_results.csv",
        index=False
    )

    pd.DataFrame(prediction_records).to_csv(
        RESULTS_DIR / "baseline_predictions.csv",
        index=False
    )

    return results_df


# ==========================================
# 7. TRAIN FIVE ML ALGORITHMS
# ==========================================

def train_models(X_dev, y_dev, df=None):

    print("\n================================")
    print("TRAINING FIVE ML ALGORITHMS")
    print("================================")

    models = get_models()

    cv = TimeSeriesSplit(n_splits=N_SPLITS)

    results = []
    grid_records = []
    prediction_records = []

    for name, pipeline in models.items():

        print(f"\nTraining {name}...")

        best_mae = float("inf")
        best_params = None

        # Hyperparameter tuning
        for params in ParameterGrid(
            PARAM_GRIDS[name]
        ):

            fold_maes = []

            for fold, (train_idx, val_idx) in enumerate(
                cv.split(X_dev), start=1
            ):

                model = clone(pipeline)
                model.set_params(**params)

                start = time.perf_counter()

                model.fit(
                    X_dev.iloc[train_idx],
                    y_dev.iloc[train_idx]
                )

                fit_seconds = (
                    time.perf_counter() - start
                )

                y_pred = model.predict(
                    X_dev.iloc[val_idx]
                )

                metrics = calculate_metrics(
                    y_dev.iloc[val_idx],
                    y_pred
                )

                fold_maes.append(
                    metrics["mae_kwh"]
                )

                grid_records.append({
                    "model": name,
                    "parameters": json.dumps(params),
                    "fold": fold,
                    "mae_kwh": metrics["mae_kwh"],
                    "rmse_kwh": metrics["rmse_kwh"],
                    "r2": metrics["r2"],
                    "fit_seconds": fit_seconds
                })

            avg_mae = float(
                np.mean(fold_maes)
            )

            if avg_mae < best_mae:
                best_mae = avg_mae
                best_params = params.copy()

        # Save best candidate fitted on development data
        best_model = clone(pipeline)
        best_model.set_params(**best_params)

        start = time.perf_counter()

        best_model.fit(X_dev, y_dev)

        refit_seconds = (
            time.perf_counter() - start
        )

        filename = (
            name.lower().replace(" ", "_")
        )

        joblib.dump(
            best_model,
            MODELS_DIR / f"{filename}.joblib"
        )

        # Record predictions for selected parameters
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_dev), start=1
        ):

            fold_model = clone(pipeline)
            fold_model.set_params(**best_params)

            fold_model.fit(
                X_dev.iloc[train_idx],
                y_dev.iloc[train_idx]
            )

            predicted = fold_model.predict(
                X_dev.iloc[val_idx]
            )

            actual = y_dev.iloc[val_idx]

            metrics = calculate_metrics(
                actual, predicted
            )

            fold_metrics.append(metrics)

            for index, true_value, pred_value in zip(
                val_idx, actual, predicted
            ):

                record = {
                    "model": name,
                    "fold": fold,
                    "row_index": int(index),
                    "actual_kwh": float(true_value),
                    "predicted_kwh": float(pred_value)
                }

                if df is not None:
                    record["target_hour"] = str(
                        df.iloc[index]["target_hour"]
                    )

                prediction_records.append(record)

        results.append({
            "model": name,
            "cv_mae_kwh": best_mae,
            "cv_rmse_kwh": float(np.mean([
                m["rmse_kwh"] for m in fold_metrics
            ])),
            "cv_r2": float(np.mean([
                m["r2"] for m in fold_metrics
            ])),
            "best_params": json.dumps(best_params),
            "refit_seconds": refit_seconds
        })

        print(
            f"{name}: "
            f"CV MAE = {best_mae:.4f} kWh"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_DIR / "training_results.csv",
        index=False
    )

    pd.DataFrame(grid_records).to_csv(
        RESULTS_DIR / "grid_search_results.csv",
        index=False
    )

    pd.DataFrame(prediction_records).to_csv(
        RESULTS_DIR / "validation_predictions.csv",
        index=False
    )

    return results_df


# ==========================================
# 8. COMPARE BASELINES AND ML MODELS
# ==========================================

def compare_models(baseline_results, ml_results):

    baseline_table = baseline_results.copy()
    baseline_table["type"] = "Baseline"

    ml_table = ml_results.copy()
    ml_table["type"] = "Machine Learning"

    comparison = pd.concat(
        [baseline_table, ml_table],
        ignore_index=True
    )

    comparison = comparison.sort_values(
        "cv_mae_kwh"
    ).reset_index(drop=True)

    comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False
    )

    print("\n================================")
    print("MODEL COMPARISON")
    print("================================")

    print(
        comparison[
            ["model", "type", "cv_mae_kwh"]
        ].to_string(index=False)
    )

    best_baseline_mae = baseline_results[
        "cv_mae_kwh"
    ].min()

    print(
        f"\nBest baseline MAE: "
        f"{best_baseline_mae:.4f} kWh"
    )

    for _, row in ml_results.iterrows():

        improvement = (
            (best_baseline_mae - row["cv_mae_kwh"])
            / best_baseline_mae
        ) * 100

        print(
            f"{row['model']} improvement "
            f"over best baseline: "
            f"{improvement:.2f}%"
        )

    return comparison


# ==========================================
# 9. MAIN EXECUTION
# ==========================================

def main():

    RESULTS_DIR.mkdir(
        parents=True, exist_ok=True
    )

    MODELS_DIR.mkdir(
        parents=True, exist_ok=True
    )

    print("================================")
    print("SMARTWATT PART 2")
    print("Baseline Forecasting & ML Training")
    print("================================")

    # Load processed Part 1 dataset
    X, y, df = load_training_data()

    # Chronological development/test split
    X_dev, X_test, y_dev, y_test = split_data(
        X, y
    )

    # Three baseline methods
    baseline_results = evaluate_baselines(
        df, X_dev, y_dev
    )

    # Five machine learning algorithms
    ml_results = train_models(
        X_dev, y_dev, df
    )

    # Compare development validation scores
    compare_models(
        baseline_results,
        ml_results
    )

    print("\n================================")
    print("PART 2 TRAINING COMPLETED!")
    print("================================")

    print("\nSaved results in:", RESULTS_DIR)
    print("Saved models in:", MODELS_DIR)

    print(
        "\nFinal test set remains untouched "
        "for Part 3 evaluation."
    )


if __name__ == "__main__":
    main()

