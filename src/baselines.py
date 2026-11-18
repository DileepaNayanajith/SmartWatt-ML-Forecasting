
import pandas as pd

BASELINE_COLUMNS = {
    "persistence": "y_lag_1h",
    "yesterday": "y_lag_24h",
    "last_week": "y_lag_168h"
}

def predict_baselines(df):
    """Generate three baseline forecasts."""

    predictions = pd.DataFrame(index=df.index)

    for name, column in BASELINE_COLUMNS.items():
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

        predictions[name] = df[column]

    return predictions

