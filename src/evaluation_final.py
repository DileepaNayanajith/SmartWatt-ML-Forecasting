"""Explicit, one-attempt final evaluation after real validation selection.

Local markers prevent accidental reruns in the same audit directory, not
malicious reuse elsewhere. Never invoke this on simulated data.
"""
import hashlib
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from src.evaluate import metrics, utc_timestamp


def lock_selection(path, *, selection, estimator, features, development_end,
                   test_start, data_provenance, feature_contract):
    """Freeze configuration before a caller opens test data.

    Caller must review real validation reports and supply an unfitted pipeline
    matching Part 2's selected configuration. No final model is saved here.
    """
    if selection.get("simulated") is not False or not data_provenance or not feature_contract:
        raise ValueError("Real validation, provenance and feature/timing contract required")
    if not utc_timestamp(development_end) < utc_timestamp(test_start):
        raise ValueError("Development must precede the test")
    if not features or len(set(features)) != len(features):
        raise ValueError("Unique ordered features required")
    record = {"selection": selection, "features": list(features),
              "development_end": development_end, "test_start": test_start,
              "data_provenance": data_provenance, "feature_contract": feature_contract,
              "estimator_repr": repr(estimator),
              "estimator_hash": joblib.hash(estimator),
              "python": platform.python_version(), "sklearn": sklearn.__version__}
    with Path(path).open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


def final_test_once(lock_path, output, estimator, development, load_test):
    """Refit development then call load_test once; output directory is the gate.

    Both frames use a timezone-aware, sorted unique DatetimeIndex, ordered
    feature columns, and actual_kwh. load_test is a callable, keeping test data
    unopened until this explicit step. On failure retain the attempt marker.
    """
    lock_path, output = Path(lock_path), Path(output)
    record = json.loads(lock_path.read_text())
    if record["selection"].get("simulated", True) or joblib.hash(estimator) != record["estimator_hash"]:
        raise ValueError("Real locked selection and identical estimator required")
    features = record["features"]
    def check(frame):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError("Timezone-aware index required")
        if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("Nonempty sorted unique rows required")
        if list(frame.drop(columns="actual_kwh").columns) != features:
            raise ValueError("Ordered feature schema mismatch")
    check(development)
    if development.index.max() > pd.Timestamp(record["development_end"]):
        raise ValueError("Development overlaps locked holdout")
    output.mkdir(parents=True, exist_ok=False)
    (output / "ATTEMPT.json").write_text(json.dumps({"lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest()}))
    fitted = clone(estimator).fit(development[features], development.actual_kwh)
    test = load_test()
    check(test)
    if test.index.min() < pd.Timestamp(record["test_start"]) or test.index.min() <= development.index.max():
        raise ValueError("Test period overlaps development")
    prediction = fitted.predict(test[features])
    scores = metrics(test.actual_kwh, prediction)
    pd.DataFrame({"actual_kwh": test.actual_kwh, "pred_kwh": prediction}).to_csv(output / "final_test_predictions.csv")
    json_scores = {key: None if pd.isna(value) else value for key, value in scores.items()}
    (output / "final_test_metrics.json").write_text(json.dumps(json_scores, allow_nan=False))
    joblib.dump(fitted, output / "smartwatt_final.joblib")
    (output / "metadata.json").write_text(json.dumps(record, indent=2))
    return scores
