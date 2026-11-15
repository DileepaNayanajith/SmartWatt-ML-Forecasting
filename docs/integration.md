# Integration notes

These are proposed handoffs to confirm together before implementation. No data
schema or model interface has been validated yet.

1. Part 1 documents the hour boundary, timezone assumptions, forecast issue time,
   target interval, units, ordered feature names and input availability. Keep
   target timestamps explicit. Only complete hourly energy totals should become
   targets; document how gaps and lag warm-up rows are handled.
2. Part 2 records exact chronological split boundaries and fold identifiers.
   Models and baselines must be compared on identical eligible target rows.
   Save preprocessing together with the estimator so the dashboard uses the
   same transformations. Record package versions and feature schema with it.
3. Proposed long-format validation file: `results/validation_predictions.csv`
   with `fold,target_timestamp,model,actual_kwh,pred_kwh`. Include each baseline
   using the same format. Add forecast issue time if needed to disambiguate
   availability. Do not insert illustrative predictions as measured results.
4. Part 3 agrees metric aggregation and high-consumption thresholds using
   development data. Keep final-test predictions/results separate from tuning
   records. Record the selected configuration before opening the final test.
5. Part 4 loads the selected pipeline plus feature/timing metadata, and replays
   only historically available inputs. Show an honest unavailable state until
   real artifacts exist. Add the actual run command once implemented.

Keep raw/processed data and model binaries local or in agreed shared storage.
Record artifact locations, versions and reproduction steps without committing
credentials. Commit modest measured result tables and plots when available.
