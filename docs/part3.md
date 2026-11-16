# Part 3 — evaluation handoff

EC5203, Group 19 EnergyScope. This branch implements evaluation infrastructure.
There are no real results, selected final models or dataset files yet. Part 1,
Part 2 and the dashboard remain their owners' placeholders. This page records
the Part 3 branch's interfaces and progress.

## Setup and checks

From the repository root (Python 3.11 recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-part3.txt
python -m pytest -q
```

The checks cover macOS ARM64 Python 3.9.6 and Python 3.11.
`requirements-part3-tested.txt` records the complete Python 3.11 environment.
Use `python -m pip install -r requirements-part3-tested.txt` to reproduce that
exact dependency set; `requirements-part3.txt` pins the direct dependencies.
The tests create deterministic SIMULATED data in memory and temporary reports.
Those are software checks, never coursework findings. No fixture is stored in
`data/` or `results/`, and no final-model binary is committed.

## Part 1 and Part 2 inputs

Part 1 must confirm target hour boundaries, dataset timezone, forecast issue
time, conversion to kWh, incomplete hours, historical availability of each
feature, and the next-hour target alignment. `rv1`/`rv2` and future-derived
features must be excluded upstream. Evaluation cannot prove that predictions
were generated without leakage merely by inspecting CSV files.

Part 2 supplies:

- `results/validation_predictions.csv`: long format columns
  `fold,target_timestamp,model,actual_kwh,pred_kwh`. Exact names:
  `Ridge`, `DecisionTree`, `RandomForest`, `GradientBoosting`, `SVR`,
  `Persistence`, `Yesterday`, `LastWeek`. One finite prediction per model per
  target, identical actual values for every model, exactly three folds.
- `results/training_targets.csv`: `fold,target_timestamp,actual_kwh` for each
  fold's complete training targets. Repeated timestamps across expanding folds
  are expected. Used to calculate the fold-specific training 90th percentile.
- A real split manifest, exact feature schema and unfitted reproducible
  pipelines (all imputation/scaling inside each pipeline), parameters, random
  seeds, training times, package versions and data provenance.

All timestamps must carry UTC offsets. Ambiguous naive timestamps are rejected;
export offset-aware timestamps explicitly. Training targets must contain all
hours in each declared training window and agree across expanding folds.
Fold identifiers must use the same type in JSON and CSV. Current hourly contract
requires complete hourly validation periods; missing-hour eligibility must be
agreed with teammates before changing this check. Never silently intersect
model outputs to hide missing predictions. Final test starts after every
validation period and is never loaded by the validation command.

Manifest structure (dates below are SCHEMA EXAMPLES, not actual dataset splits):

```json
{
  "simulated": true,
  "calendar_timezone": "Europe/Brussels",
  "test_start": "2024-01-06T00:00:00Z",
  "folds": [
    {"fold": 1, "train_start": "2024-01-01T00:00:00Z", "train_end": "2024-01-02T23:00:00Z", "validation_start": "2024-01-03T00:00:00Z", "validation_end": "2024-01-03T23:00:00Z"},
    {"fold": 2, "train_start": "2024-01-01T00:00:00Z", "train_end": "2024-01-03T23:00:00Z", "validation_start": "2024-01-04T00:00:00Z", "validation_end": "2024-01-04T23:00:00Z"},
    {"fold": 3, "train_start": "2024-01-01T00:00:00Z", "train_end": "2024-01-04T23:00:00Z", "validation_start": "2024-01-05T00:00:00Z", "validation_end": "2024-01-05T23:00:00Z"}
  ]
}
```

Set `simulated: false` only for genuine project data. Confirm the timezone from
source documentation; the example timezone must not become an unverified data
assumption. Forecast next-hour baselines at issue time using the last completed
hour, target-hour-minus-24-hours and target-hour-minus-168-hours respectively.
Part 2 owns those implementations and lag warm-up eligibility.

## Generate validation reports

After receiving the real files, run from the repository root:

```bash
python -m src.evaluate \
  --predictions results/validation_predictions.csv \
  --manifest results/splits.json \
  --training-targets results/training_targets.csv \
  --output results/part3-validation-v1
```

A new output directory is required so previous evidence is preserved. Reports:

- Per-fold MAE/RMSE/R² and sample counts; equal-fold metric means and sample
  standard deviations; pooled metrics in a separate table.
- Best naive baseline chosen on mean validation MAE; percentage MAE improvement
  uses this same equal-fold aggregation. Zero baseline MAE gives undefined
  improvement, written as an empty CSV cell. R² is empty for constant targets
  or fewer than two observations.
- High-consumption scores and signed bias per fold using training 90th
  percentiles (`actual >= threshold`); hours/weekdays use the manifest timezone.
  High-consumption folds with no events have no row; do not treat them as zero
  error. Error tables include counts to reveal sparse groups.
- MAE/RMSE/R² comparison plots, fold-specific actual/predicted time lines,
  scatter plot, hourly/weekday error plots, checked predictions, input SHA256
  hashes, and `selection.json`.

Selection uses mean validation MAE, then mean RMSE, then name for deterministic
exact ties. It records a validation candidate, not a final-model claim. Review
fold stability, high-consumption bias, training/prediction cost and robustness
before locking. If a near-tie changes the decision for cost reasons, record that
rule and decision before test access. Do not choose the baseline on test scores.

## Importance, ablations and optimization

`src/evaluation_experiments.py` accepts numeric feature frames with a sorted,
unique, timezone-aware DatetimeIndex. The target column is `actual_kwh`.

```python
from src.evaluation_experiments import run_ablations, importance, robustness

# pipelines is supplied by Part 2: unfitted sklearn pipelines, keyed by name.
# groups contains explicitly reviewed feature lists from Part 1.
# A: history; B: history+calendar; C: B+temperature/humidity;
# D: C+lighting+historically available weather.
experiments = run_ablations(data, manifest, pipelines, groups)
experiments.to_csv("results/ablations.csv", index=False)
```

Every model/feature-set combination is freshly cloned and fitted per fold;
preprocessing stays inside the fold. Named parameter variants may be passed in
`pipelines` for validation-based optimization. Keep the search space modest,
record parameters/seeds and compare equal-fold means. This module does not
replace Part 2's five model implementations or prescribe unvalidated settings.

`importance(fitted_pipeline, X_validation, y_validation)` returns deterministic
permutation MAE increases and repeat standard deviations. Call separately per
fold on the corresponding fitted pipeline and validation set, then save the
fold identifier. Correlated features complicate interpretation; importance is
reliance, not causation. Random permutations disrupt temporal structure, so
interpret results as diagnostics rather than causal evidence.

`robustness(fitted_pipeline, X_train, X_validation, y_validation,
missing_groups, delayed_groups)` compares normal inputs with missing sensor
groups and delayed inputs. Dictionaries map scenario names to feature lists;
include temperature, humidity and multiple-sensor scenarios explicitly. Default
missing rate is 10%, fixed seed 42, delay one hour. Delay looks up exact earlier
timestamps from training plus validation history, never interpolates from the
future. Pass the feature rows available at forecast issue time. Keep historical
context contiguous where possible. Failure to handle NaNs is a reported failed
scenario; an imputer must already have been fitted within training. Save outputs
per fold with corruption rate, delay, seed and parameters. Targets never change.

## Locked final evaluation — only when real dependencies exist

`src/evaluation_final.py` exposes two explicit steps, not part of the validation
CLI. Do not call them now: real inputs are missing.

1. Review real validation evidence and call `lock_selection` with its selection
   record, exact unfitted pipeline, ordered feature names, offset-aware
   development end/test start, data provenance and the agreed feature/timing
   contract. It writes an exclusive JSON lock containing pipeline configuration
   hash, versions and decision metadata. Review and keep that record before
   loading final-test data.
2. Call `final_test_once(lock_path, output_dir, estimator, development,
   load_test)` with the identical pipeline configuration, full development
   feature/target frame, and a callable that opens the untouched test data.
   It refits the whole pipeline on development only, invokes the loader after
   creating an attempt marker, validates chronology/schema, scores once and
   saves the real fitted pipeline plus metadata and final predictions.

The output directory cannot already exist. Failed attempts retain the marker;
inspect failures and document any recovery, never quietly rerun test-driven
optimization. This is a local accidental-reuse guard, not a global access
control system. The caller owns honest provenance, true untouched-test status,
complete development data and alignment with Part 2's chosen model. Baseline
comparison on final test can use only the already locked validation-selected
baseline; no test-set reselection is permitted. The current final helper reports
selected-model MAE/RMSE/R²; Part 2 must also supply the locked baseline predictions
for the final report's baseline-improvement calculation.

Part 4 should use the saved full pipeline and metadata only after this real
step succeeds. Never provide a simulated model to the dashboard as a final
SmartWatt artifact.

## Verification record — 16 November 2026

18 tests passed on Python 3.11.17, macOS ARM64. Checks cover known metrics,
undefined R²/zero-baseline handling, alignment and chronology rejection, training
threshold leakage, actual-target consistency, report generation and CLI input
hashes, cloned ablation training, repeatable permutation importance, exact
past-only delayed inputs, simulated-final rejection and final-attempt guards.
A comparison plot was visually checked for readable labels and explicit
SIMULATED marking. The 14 reported warnings are Matplotlib/pyparsing dependency
deprecations; no test failed. `git diff --check` passed.

Pending real inputs: processed hourly data/feature timing contract from Part 1;
three-fold manifest, complete training targets, all eight prediction series,
unfitted/fitted pipelines, parameters/seeds and training times from Part 2.
Actual optimization experiments, model selection, final-baseline comparison and
untouched final evaluation remain pending those dependencies.
