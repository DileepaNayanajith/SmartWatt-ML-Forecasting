"""SIMULATED fixtures only: these values are not measured SmartWatt results."""
import numpy as np
import pandas as pd
import pytest
from src.evaluate import ALL_MODELS


@pytest.fixture
def simulated():
    index = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"energy_lag": rng.uniform(.1, .8, len(index)),
                         "temperature": rng.normal(20, 3, len(index))}, index=index)
    data["actual_kwh"] = .8 * data.energy_lag + .01 * data.temperature
    manifest = {"simulated": True, "calendar_timezone": "Europe/Brussels",
                "test_start": str(index[112]), "folds": []}
    predictions, training = [], []
    for fold, start in enumerate((40, 64, 88), 1):
        manifest["folds"].append({"fold": fold, "train_start": str(index[0]),
                                  "train_end": str(index[start-1]),
                                  "validation_start": str(index[start]),
                                  "validation_end": str(index[start+23])})
        for timestamp in index[:start]:
            training.append({"fold": fold, "target_timestamp": timestamp,
                             "actual_kwh": data.loc[timestamp, "actual_kwh"]})
        for rank, model in enumerate(ALL_MODELS):
            for timestamp in index[start:start+24]:
                actual = data.loc[timestamp, "actual_kwh"]
                predictions.append({"fold": fold, "target_timestamp": timestamp, "model": model,
                                    "actual_kwh": actual, "pred_kwh": actual + .01*(rank+1)})
    return data, manifest, pd.DataFrame(predictions), pd.DataFrame(training)
