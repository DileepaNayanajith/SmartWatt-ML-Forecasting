"""Part 3 experiments. Part 2 supplies unfitted sklearn-compatible pipelines.

All preprocessing must live inside each pipeline. Never pass a transformer
already fitted on the complete dataset. These helpers do not save models.
"""
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from src.evaluate import metrics, validate_folds


def fold_data(data, manifest, features, target="actual_kwh"):
    """Yield chronological training/validation slices; exclude final-test rows."""
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:
        raise ValueError("Use a timezone-aware DatetimeIndex")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("Rows must be unique and chronological")
    if target in features or not features or len(set(features)) != len(features):
        raise ValueError("Features must be unique and exclude the target")
    for f in validate_folds(manifest).itertuples():
        train = data.loc[f.train_start:f.train_end]
        valid = data.loc[f.validation_start:f.validation_end]
        if train.empty or not valid.index.equals(pd.date_range(f.validation_start, f.validation_end, freq="h")):
            raise ValueError("Missing training or hourly validation rows")
        yield f.fold, train[features], train[target], valid[features], valid[target]


def run_ablations(data, manifest, estimators, feature_groups):
    """Retrain each candidate for each feature set on the exact same folds."""
    rows = []
    for experiment, features in feature_groups.items():
        for fold, xt, yt, xv, yv in fold_data(data, manifest, features):
            for name, estimator in estimators.items():
                fitted = clone(estimator)
                start = time.perf_counter(); fitted.fit(xt, yt)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter(); predicted = fitted.predict(xv)
                rows.append({"experiment": experiment, "model": name, "fold": fold,
                             "fit_seconds": fit_seconds, "predict_seconds": time.perf_counter()-start,
                             **metrics(yv, predicted)})
    return pd.DataFrame(rows)


def importance(fitted_pipeline, x_validation, y_validation, repeats=5, seed=42):
    """Validation MAE increase after permutation; reliance, not causal importance."""
    result = permutation_importance(fitted_pipeline, x_validation, y_validation,
                                    scoring="neg_mean_absolute_error", n_repeats=repeats,
                                    random_state=seed)
    return pd.DataFrame({"feature": x_validation.columns,
                         "MAE_increase_mean": result.importances_mean,
                         "MAE_increase_std": result.importances_std}).sort_values("MAE_increase_mean", ascending=False)


def robustness(fitted_pipeline, x_train, x_validation, y_validation,
               missing_groups, delayed_groups, missing_fraction=.1, delay_hours=1, seed=42):
    """Fixed fitted pipeline; corrupt validation inputs only, using past observations.

    x_train supplies historical context for delayed features. Missing-group names
    map to column lists. Delay uses exact timestamp lookup, never future filling.
    Unsupported missing inputs are reported as failed scenarios, not omitted.
    """
    if not 0 < missing_fraction <= 1 or delay_hours < 1:
        raise ValueError("Invalid corruption settings")
    history = pd.concat([x_train, x_validation])
    if history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("Training history must precede validation, with unique timestamps")
    rng = np.random.default_rng(seed)
    scenarios = {"normal": x_validation.copy()}
    for name, columns in missing_groups.items():
        changed = x_validation.copy()
        count = max(1, int(np.ceil(len(changed) * missing_fraction)))
        positions = rng.choice(len(changed), size=count, replace=False)
        changed.loc[changed.index[positions], columns] = np.nan
        scenarios[f"missing_{name}"] = changed
    for name, columns in delayed_groups.items():
        changed = x_validation.copy()
        past = history.reindex(changed.index - pd.Timedelta(hours=delay_hours))
        changed.loc[:, columns] = past[columns].to_numpy()
        scenarios[f"delayed_{name}"] = changed
    rows = []
    for name, changed in scenarios.items():
        try:
            rows.append({"scenario": name, "status": "ok", **metrics(y_validation, fitted_pipeline.predict(changed))})
        except (ValueError, TypeError) as exc:
            rows.append({"scenario": name, "status": "failed", "error": str(exc)})
    result = pd.DataFrame(rows)
    normal = result.loc[result.scenario == "normal", "MAE"].iloc[0] if "MAE" in result else np.nan
    result["MAE_change"] = result.get("MAE", np.nan) - normal
    return result
