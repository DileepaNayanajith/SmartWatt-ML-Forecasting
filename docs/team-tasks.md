# Team tasks

Assign one member to each part after agreeing within the group. These are planned
tasks, not completed work. Initial exploration should be brought into this repo
rather than repeated unnecessarily.

## Part 1 — Preprocessing and features

Branch: `part-1-preprocessing`

- Verify timestamps, Wh units, duplicates, gaps and sensor quality.
- Aggregate complete hours, convert Wh to kWh and define the next-hour target.
- Document incomplete-hour handling and forecast-time availability.
- Build energy lags at 1, 2, 3, 6, 12, 24 and 168 hours, rolling summaries,
  historical changes, calendar and available environmental features.
- Exclude random columns and prevent future-data leakage.
- Deliver `src/preprocessing.py`, `src/features.py`, the local processed dataset
  and feature definitions, including enough features for the course requirement.

## Part 2 — Baselines and training

Branch: `part-2-training`

- Agree on development/final-test boundaries and three expanding-window folds.
- Implement persistence, yesterday and last-week baselines.
- Train Ridge, Decision Tree, Random Forest, Gradient Boosting and SVR.
- Fit scaling/imputation within training folds and tune on development data only.
- Record parameters, seeds, training times and validation predictions.
- Deliver `src/baselines.py`, `src/train.py`, a training notebook, candidate
  pipelines and reproducible experiment records.

## Part 3 — Evaluation and optimization

Branch: `part-3-evaluation`

- Compare the same validation timestamps using MAE, RMSE and R².
- Report improvement over the development-selected best naive baseline.
- Produce comparison tables, plots and actual-versus-predicted graphs.
- Analyze high-consumption, hourly and weekday errors, feature importance,
  feature-group ablation and missing/delayed-input robustness.
- Select using validation performance and computational cost; coordinate any
  retraining with Part 2. Lock the configuration, refit on development data and
  evaluate the untouched final test once.
- Deliver `src/evaluate.py`, measured results, the final pipeline and test report.

## Part 4 — Dashboard and integration

Branch: `part-4-dashboard`

- Build `dashboard/app.py` around the selected pipeline and agreed feature schema.
- Replay historical next-hour forecasts with timing, model name and kWh units.
- Show actual energy where available, errors, plots and measured metrics.
- Handle invalid, missing or delayed inputs consistently with training.
- Integrate the four parts, test the complete application and document setup.

All members contribute to the final report and presentation. Each member should
commit their own actual work; branch names do not establish authorship.
