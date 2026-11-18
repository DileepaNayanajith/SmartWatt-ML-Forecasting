
import numpy as np
import pandas as pd
import pytest

from src.baselines import predict_baselines
from src.models import get_models
from src.train import (
    load_training_data,
    split_data,
    calculate_metrics
)

def test_three_baselines():
    df = pd.DataFrame({
        "y_lag_1h": [0.2, 0.3],
        "y_lag_24h": [0.4, 0.5],
        "y_lag_168h": [0.6, 0.7]
    })

    predictions = predict_baselines(df)

    assert predictions.shape == (2, 3)
    assert predictions["persistence"].iloc[0] == 0.2
    assert predictions["yesterday"].iloc[0] == 0.4
    assert predictions["last_week"].iloc[0] == 0.6


def test_missing_baseline_feature():
    df = pd.DataFrame({
        "y_lag_1h": [0.2]
    })

    with pytest.raises(ValueError):
        predict_baselines(df)


def test_five_models():
    models = get_models()

    assert len(models) == 5
    assert set(models) == {
        "Ridge",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVR"
    }


def test_dataset_and_split():
    X, y, df = load_training_data()

    assert len(df) == 3121
    assert X.shape[1] == 40

    X_dev, X_test, y_dev, y_test = split_data(X, y)

    assert len(X_dev) == 2497
    assert len(X_test) == 624
    assert len(y_dev) == len(X_dev)
    assert len(y_test) == len(X_test)

    assert (
        df["target_hour"].iloc[len(X_dev)]
        > df["target_hour"].iloc[len(X_dev) - 1]
    )


def test_metrics():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 2.0])

    metrics = calculate_metrics(actual, predicted)

    assert metrics["mae_kwh"] == pytest.approx(1 / 3)
    assert metrics["rmse_kwh"] == pytest.approx(
        np.sqrt(1 / 3)
    )
    assert metrics["r2"] == pytest.approx(0.5)
