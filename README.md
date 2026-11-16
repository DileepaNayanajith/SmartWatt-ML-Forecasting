# SmartWatt

Next-hour appliance energy forecasting using classical machine learning.

EC5203 — Machine Learning, University of Ruhuna

Group 19 — EnergyScope

## Status

On `part-3-evaluation`, the evaluation module, experiment helpers and simulated
software tests are implemented. See [Part 3 setup and handoff](docs/part3.md).
Preprocessing, model training and the dashboard are still placeholders. There
are no real trained models or measured forecasting results in this repository.

The project handover says initial exploration has started in Jupyter, including
data inspection and a correlation heatmap. Those notebooks were not available
when this structure was created and still need to be added by the team.

## Project scope

Use the [UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
(`energydata_complete.csv`) to forecast total appliance energy during the next
hour in kWh. The handover describes 10-minute readings from one Belgian house.
Verify timestamp meaning and energy units before hourly aggregation.

The planned models are Ridge Regression, Decision Tree, Random Forest,
Gradient Boosting and SVR. Compare them with persistence, same-hour-yesterday
and same-hour-last-week baselines. The application will demonstrate historical
forecast replay; it is not a live electricity monitoring system.

## Layout

```text
data/raw/          Original dataset, stored locally
data/processed/    Generated hourly data, stored locally
src/               Preprocessing, features, baselines, training and evaluation
notebooks/         Existing exploration and later experiment notebooks
models/            Generated pipelines, stored locally
results/           Measured tables, plots and experiment records
dashboard/        Streamlit application placeholder
docs/             Team tasks and integration notes
```

## Local setup

Use Python 3.11 or newer. The dependency list is an initial set, not a tested or
locked environment. Record tested versions when implementation begins.

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Place the dataset at `data/raw/energydata_complete.csv`. Data and generated model
files are ignored by Git. Record the source, download date and any cleaning
choices when preparing the data. There is no training or dashboard command yet;
add execution instructions as each part becomes usable.

## Four-member workflow

The team can allocate one member to each part. Names are deliberately left open.
All four branches start from the same scaffold on `main`.

| Part | Branch | Main responsibility |
| --- | --- | --- |
| 1 | `part-1-preprocessing` | Hourly data, next-hour target, features and leakage checks |
| 2 | `part-2-training` | Chronological folds, three baselines and five ML models |
| 3 | `part-3-evaluation` | Fair comparison, error analysis and final model selection |
| 4 | `part-4-dashboard` | Historical replay dashboard, integration and testing |

See [team tasks](docs/team-tasks.md) for deliverables and
[integration notes](docs/integration.md) for the handoffs to agree before coding.

Clone the repository, then switch to your assigned branch:

```bash
git clone https://github.com/DileepaNayanajith/SmartWatt-ML-Forecasting.git
cd SmartWatt-ML-Forecasting
git switch part-1-preprocessing
```

Commit your actual work on your branch and open a pull request into `main` for
review. After shared changes are merged, bring the updated `main` into your
branch before continuing. Add teammates through repository Settings →
Collaborators using their GitHub usernames.

## Evaluation rules

Keep observations in time order. Fit learned preprocessing within training
folds, use only inputs available at forecast time, and reserve an untouched
chronological final-test period. Do not tune against final-test results.
Exclude `rv1` and `rv2`, and check weather availability and interpolation for
leakage. Use classical ML only, as required by the handover.

Report actual experiments, including limitations. Do not name a winning model
until validation has been completed. The final report and presentation are
shared team responsibilities.
