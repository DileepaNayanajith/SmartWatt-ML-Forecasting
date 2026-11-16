"""Part 3: strict validation of Part 2 predictions and reproducible reports.

No training, data download, or final-test access happens on import.
See docs/part3.md for the handoff contract.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODELS = ("Ridge", "DecisionTree", "RandomForest", "GradientBoosting", "SVR")
BASELINES = ("Persistence", "Yesterday", "LastWeek")
ALL_MODELS = MODELS + BASELINES
KEY = ["fold", "target_timestamp"]


def metrics(actual, predicted):
    """Errors in kWh; R² is undefined for fewer than two or constant targets."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("Expected nonempty, equally sized one-dimensional targets")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Targets and predictions must be finite")
    return {"MAE": mean_absolute_error(actual, predicted),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": r2_score(actual, predicted) if len(actual) > 1 and np.ptp(actual) > 0 else np.nan,
            "n": len(actual)}


def utc_timestamp(value):
    """Reject ambiguous local times instead of silently assuming UTC."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Every timestamp must include a UTC offset/timezone")
    return timestamp.tz_convert("UTC")


def utc_series(values):
    return pd.to_datetime(values.map(utc_timestamp), utc=True)


def validate_folds(manifest):
    """Check three disjoint validations with a fixed-start, growing training window."""
    if not isinstance(manifest.get("simulated"), bool):
        raise ValueError("Manifest must explicitly mark simulated true or false")
    folds = pd.DataFrame(manifest["folds"]).copy()
    if len(folds) != 3 or folds.fold.duplicated().any():
        raise ValueError("Exactly three distinct folds are required")
    for col in ("train_start", "train_end", "validation_start", "validation_end"):
        folds[col] = utc_series(folds[col])
    folds = folds.sort_values("validation_start").reset_index(drop=True)
    if folds.isna().any().any():
        raise ValueError("Fold metadata cannot contain missing values")
    test_start = utc_timestamp(manifest["test_start"])
    for i, f in folds.iterrows():
        if not f.train_start <= f.train_end < f.validation_start <= f.validation_end < test_start:
            raise ValueError("Invalid chronological boundaries or test overlap")
        if i and not (f.train_start == folds.iloc[0].train_start
                      and f.train_end > folds.iloc[i-1].train_end
                      and f.validation_start > folds.iloc[i-1].validation_end
                      and f.train_end >= folds.iloc[i-1].validation_end):
            raise ValueError("Expected chronological expanding-window folds")
    return folds


def validate_predictions(predictions, manifest):
    """Reject partial comparisons rather than silently dropping unmatched rows."""
    folds = validate_folds(manifest)
    frame = predictions.copy()
    required = KEY + ["model", "actual_kwh", "pred_kwh"]
    if not set(required) <= set(frame):
        raise ValueError(f"Required columns: {required}")
    frame["target_timestamp"] = utc_series(frame.target_timestamp)
    if frame[required].isna().any().any() or frame.duplicated(KEY + ["model"]).any():
        raise ValueError("Missing values or duplicate prediction keys")
    if set(frame.model) != set(ALL_MODELS) or set(frame.fold) != set(folds.fold):
        raise ValueError("All five models, three baselines, and three folds are required")
    metrics(frame.actual_kwh, frame.pred_kwh)
    if (frame.actual_kwh < 0).any():
        raise ValueError("Actual kWh cannot be negative")
    reference = frame[frame.model == MODELS[0]].set_index(KEY).sort_index().actual_kwh
    if reference.reset_index().target_timestamp.duplicated().any():
        raise ValueError("Validation targets cannot overlap across folds")
    for name, group in frame.groupby("model"):
        actual = group.set_index(KEY).sort_index().actual_kwh
        if not actual.index.equals(reference.index) or not np.array_equal(actual.values, reference.values):
            raise ValueError(f"Target rows/values differ for {name}")
    for f in folds.itertuples():
        times = frame.loc[frame.fold == f.fold, "target_timestamp"]
        if not times.between(f.validation_start, f.validation_end).all():
            raise ValueError("Predictions outside declared validation period")
        expected = pd.date_range(f.validation_start, f.validation_end, freq="h")
        if not pd.DatetimeIndex(times.unique()).sort_values().equals(expected):
            raise ValueError("Expected every hourly target; agree gap eligibility before adapting this contract")
    return frame.sort_values(KEY + ["model"])


def _scores(frame, groups):
    rows = []
    for key, group in frame.groupby(groups, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(groups, key), **metrics(group.actual_kwh, group.pred_kwh)))
    return pd.DataFrame(rows)


def evaluate(predictions, manifest, training_targets):
    """Return tables and a validation-only decision; never read final-test data.

    training_targets: fold,target_timestamp,actual_kwh for each fold's training
    targets. Targets are checked against the declared training interval.
    """
    frame = validate_predictions(predictions, manifest)
    folds = validate_folds(manifest)
    train = training_targets.copy()
    train.target_timestamp = utc_series(train.target_timestamp)
    if set(train.fold) != set(folds.fold) or train.duplicated(KEY).any() or train.isna().any().any():
        raise ValueError("Training targets require exactly the declared folds and unique keys")
    metrics(train.actual_kwh, train.actual_kwh)
    thresholds = {}
    for f in folds.itertuples():
        group = train[train.fold == f.fold]
        if group.empty or not group.target_timestamp.between(f.train_start, f.train_end).all():
            raise ValueError("Threshold targets must come only from the respective training interval")
        if (group.actual_kwh < 0).any():
            raise ValueError("Negative training energy")
        expected = pd.date_range(f.train_start, f.train_end, freq="h")
        if not pd.DatetimeIndex(group.target_timestamp).sort_values().equals(expected):
            raise ValueError("Training thresholds require all hourly training targets")
        thresholds[f.fold] = group.actual_kwh.quantile(.9)
    known_targets = pd.concat([train[KEY + ["actual_kwh"]], frame[KEY + ["actual_kwh"]]])
    if known_targets.groupby("target_timestamp").actual_kwh.nunique().gt(1).any():
        raise ValueError("Actual targets disagree between training/validation folds")
    fold_scores = _scores(frame, ["model", "fold"])
    summary = fold_scores.groupby("model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    baseline = summary[summary.model.isin(BASELINES)].sort_values(["MAE_mean", "RMSE_mean", "model"]).iloc[0]
    summary["baseline"] = baseline.model
    summary["MAE_improvement_pct"] = (100 * (baseline.MAE_mean - summary.MAE_mean) / baseline.MAE_mean
                                        if baseline.MAE_mean > 0 else np.nan)
    winner = summary[summary.model.isin(MODELS)].sort_values(["MAE_mean", "RMSE_mean", "model"]).iloc[0].model
    frame["high_threshold_kwh"] = frame.fold.map(thresholds)
    frame["high_consumption"] = frame.actual_kwh >= frame.high_threshold_kwh
    # Calendar timezone must be agreed by Part 1, rather than inferred from the machine.
    local = frame.target_timestamp.dt.tz_convert(manifest["calendar_timezone"])
    frame["hour"], frame["weekday"] = local.dt.hour, local.dt.dayofweek
    frame["signed_error_kwh"] = frame.pred_kwh - frame.actual_kwh
    high = frame[frame.high_consumption]
    high_scores = _scores(high, ["model", "fold"]) if len(high) else pd.DataFrame()
    if len(high):
        high_scores = high_scores.merge(high.groupby(["model", "fold"]).signed_error_kwh.mean()
                                       .rename("bias_kwh").reset_index())
    tables = {"validation_fold_scores": fold_scores, "validation_summary": summary,
              "validation_pooled_scores": _scores(frame, ["model"]),
              "high_consumption": high_scores,
              "hourly_errors": _scores(frame, ["model", "hour"]),
              "weekday_errors": _scores(frame, ["model", "weekday"]),
              "validation_predictions_checked": frame}
    decision = {"selected_model": winner, "selected_baseline": baseline.model,
                "criterion": "equal-fold mean MAE, then mean RMSE, then name",
                "status": "validation candidate only; not a final model",
                "simulated": bool(manifest.get("simulated", False)),
                "high_thresholds_kwh": {str(k): v for k, v in thresholds.items()}}
    return tables, decision


def write_report(tables, decision, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "selection.json").write_text(json.dumps(decision, indent=2, allow_nan=False))
    label = "SIMULATED — software test only" if decision["simulated"] else "Development validation"
    summary = tables["validation_summary"].set_index("model")
    for metric in ("MAE", "RMSE", "R2"):
        values = summary[f"{metric}_mean"].dropna()
        if len(values):
            ax = values.plot.bar(yerr=summary.loc[values.index, f"{metric}_std"].fillna(0), capsize=3)
        else:
            _, ax = plt.subplots()
            ax.text(.5, .5, "Metric undefined for these targets", ha="center")
        ax.set(title=f"{label}: {metric} (mean ± fold SD)", ylabel=metric)
        plt.tight_layout(); plt.savefig(output / f"comparison_{metric}.png"); plt.close()
    frame = tables["validation_predictions_checked"]
    selected = frame[frame.model == decision["selected_model"]]
    for fold, group in selected.groupby("fold"):
        ax = group.plot(x="target_timestamp", y=["actual_kwh", "pred_kwh"], figsize=(10, 4))
        ax.set(title=f"{label}: {decision['selected_model']}, fold {fold}", ylabel="kWh")
        plt.tight_layout(); plt.savefig(output / f"actual_predicted_fold_{fold}.png"); plt.close()
    ax = selected.plot.scatter(x="actual_kwh", y="pred_kwh")
    bounds = [min(selected.actual_kwh.min(), selected.pred_kwh.min()),
              max(selected.actual_kwh.max(), selected.pred_kwh.max())]
    ax.plot(bounds, bounds, "k--"); ax.set_title(label)
    plt.tight_layout(); plt.savefig(output / "actual_predicted_scatter.png"); plt.close()
    for name, column in (("hourly_errors", "hour"), ("weekday_errors", "weekday")):
        tables[name].pivot(index=column, columns="model", values="MAE").plot(figsize=(10, 4), marker=".")
        plt.title(label); plt.ylabel("MAE (kWh)"); plt.tight_layout()
        plt.savefig(output / f"{name}.png"); plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--training-targets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    tables, decision = evaluate(pd.read_csv(args.predictions), manifest, pd.read_csv(args.training_targets))
    write_report(tables, decision, args.output)
    paths = {key: getattr(args, key) for key in ("predictions", "manifest", "training_targets")}
    hashes = {key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key, path in paths.items()}
    (Path(args.output) / "input_hashes.json").write_text(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
