"""All experiments here are simulated software checks, not project findings."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from src.evaluate import metrics, evaluate, validate_predictions, write_report
from src.evaluation_experiments import run_ablations, importance, robustness
from src.evaluation_final import lock_selection, final_test_once


def test_known_metrics():
    score = metrics([1, 2, 3], [2, 2, 2])
    assert score["MAE"] == pytest.approx(2/3)
    assert score["RMSE"] == pytest.approx(np.sqrt(2/3))
    assert score["R2"] == 0
    assert np.isnan(metrics([1, 1], [1, 1])["R2"])
    with pytest.raises(ValueError):
        metrics([1, np.nan], [1, 2])


def test_report_and_selection(simulated, tmp_path):
    _, manifest, predictions, training = simulated
    tables, decision = evaluate(predictions.sample(frac=1, random_state=7), manifest, training)
    assert decision["selected_model"] == "Ridge"
    assert decision["selected_baseline"] == "Persistence"
    assert tables["validation_summary"].set_index("model").loc["Ridge", "MAE_mean"] == pytest.approx(.01)
    assert tables["validation_summary"].set_index("model").loc["Ridge", "MAE_improvement_pct"] == pytest.approx(100*5/6)
    write_report(tables, decision, tmp_path / "SIMULATED")
    assert len(list((tmp_path / "SIMULATED").glob("*.png"))) == 9
    assert json.loads((tmp_path / "SIMULATED/selection.json").read_text())["simulated"]
    with pytest.raises(FileExistsError):
        write_report(tables, decision, tmp_path / "SIMULATED")


@pytest.mark.parametrize("problem", ["missing", "duplicate", "actual", "nonfinite", "test", "fold"])
def test_reject_unfair_comparisons(simulated, problem):
    _, manifest, predictions, _ = simulated
    if problem == "missing": predictions = predictions.iloc[1:]
    if problem == "duplicate": predictions = pd.concat([predictions, predictions.iloc[:1]])
    if problem == "actual": predictions.loc[0, "actual_kwh"] += 1
    if problem == "nonfinite": predictions.loc[0, "pred_kwh"] = np.inf
    if problem == "test": manifest["test_start"] = manifest["folds"][-1]["validation_end"]
    if problem == "fold": manifest["folds"][1]["train_start"] = manifest["folds"][0]["train_end"]
    with pytest.raises(ValueError):
        validate_predictions(predictions, manifest)


def test_training_threshold_leakage(simulated):
    _, manifest, predictions, training = simulated
    training.loc[0, "target_timestamp"] = pd.Timestamp(manifest["test_start"])
    with pytest.raises(ValueError): evaluate(predictions, manifest, training)


def test_experiments(simulated):
    data, manifest, _, _ = simulated
    estimator = make_pipeline(SimpleImputer(), Ridge())
    results = run_ablations(data, manifest, {"Ridge": estimator},
                            {"history": ["energy_lag"], "all": ["energy_lag", "temperature"]})
    assert len(results) == 6
    assert not hasattr(estimator[-1], "coef_")  # cloned, never fits supplied estimator
    train, valid = data.iloc[:40], data.iloc[40:64]
    features = ["energy_lag", "temperature"]
    fitted = estimator.fit(train[features], train.actual_kwh)
    first = importance(fitted, valid[features], valid.actual_kwh)
    pd.testing.assert_frame_equal(first, importance(fitted, valid[features], valid.actual_kwh))
    result = robustness(fitted, train[features], valid[features], valid.actual_kwh,
                        {"temperature": ["temperature"], "multiple": features},
                        {"weather": ["temperature"]})
    assert len(result) == 4
    assert result.status.eq("ok").all()
    assert result.iloc[0].MAE_change == 0


def test_final_gate_rejects_simulation(simulated, tmp_path):
    _, manifest, predictions, training = simulated
    _, decision = evaluate(predictions, manifest, training)
    with pytest.raises(ValueError, match="Real validation"):
        lock_selection(tmp_path/"lock.json", selection=decision, estimator=Ridge(),
                       features=["energy_lag"], development_end="2024-01-01T00:00:00Z",
                       test_start="2024-01-02T00:00:00Z", data_provenance="SIMULATED", feature_contract="test")


def test_final_once_control_flow_with_mock_attestation(simulated, tmp_path):
    # Exercise gate mechanics with an explicitly mocked real-input attestation.
    # Nothing from this test may be reported as a real final model or score.
    data, _, _, _ = simulated
    estimator = make_pipeline(SimpleImputer(), Ridge())
    development, test = data.iloc[:112], data.iloc[112:]
    lock = tmp_path/"lock.json"
    lock_selection(lock, selection={"simulated": False, "selected_model": "Ridge"},
                   estimator=estimator, features=["energy_lag", "temperature"],
                   development_end=str(development.index[-1]), test_start=str(test.index[0]),
                   data_provenance="MOCK ATTESTATION — simulated test fixture", feature_contract="SIMULATED")
    calls = []
    def loader():
        calls.append(1)
        return test
    final_test_once(lock, tmp_path/"attempt", estimator, development, loader)
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        final_test_once(lock, tmp_path/"attempt", estimator, development, loader)
    assert len(calls) == 1


def test_naive_time_and_incomplete_training_rejected(simulated):
    _, manifest, predictions, training = simulated
    naive = copy.deepcopy(manifest)
    naive["test_start"] = "2024-01-06"
    with pytest.raises(ValueError, match="timezone"):
        validate_predictions(predictions, naive)
    with pytest.raises(ValueError, match="all hourly"):
        evaluate(predictions, manifest, training.iloc[1:])


def test_inconsistent_history_rejected(simulated):
    _, manifest, predictions, training = simulated
    training.loc[0, "actual_kwh"] += .1
    with pytest.raises(ValueError, match="disagree"):
        evaluate(predictions, manifest, training)


def test_delayed_weather_uses_exact_past_values(simulated):
    data, _, _, _ = simulated
    features = ["energy_lag", "temperature"]
    train, valid = data.iloc[:40], data.iloc[40:64]
    captured = []
    class Recorder:
        def predict(self, frame):
            captured.append(frame.copy())
            return np.zeros(len(frame))
    robustness(Recorder(), train[features], valid[features], valid.actual_kwh,
               {}, {"weather": ["temperature"]})
    expected = data.temperature.reindex(valid.index - pd.Timedelta(hours=1)).to_numpy()
    np.testing.assert_array_equal(captured[1].temperature, expected)
    np.testing.assert_array_equal(captured[1].energy_lag, valid.energy_lag)


def test_failed_final_fit_never_opens_test(simulated, tmp_path):
    data, _, _, _ = simulated
    estimator = Ridge()
    dev = data.iloc[:112].copy()
    lock = tmp_path/"lock.json"
    lock_selection(lock, selection={"simulated": False, "selected_model": "Ridge"},
                   estimator=estimator, features=["energy_lag", "temperature"],
                   development_end=str(dev.index[-1]), test_start=str(data.index[112]),
                   data_provenance="MOCK: SIMULATED", feature_contract="SIMULATED")
    dev.loc[dev.index[0], "energy_lag"] = np.nan
    def never_open():
        pytest.fail("Test opened before successful development refit")
    with pytest.raises(ValueError):
        final_test_once(lock, tmp_path/"attempt", estimator, dev, never_open)
    assert (tmp_path/"attempt/ATTEMPT.json").exists()


def test_validation_cli(simulated, tmp_path):
    import subprocess
    import sys
    _, manifest, predictions, training = simulated
    predictions.to_csv(tmp_path/"predictions.csv", index=False)
    training.to_csv(tmp_path/"training.csv", index=False)
    (tmp_path/"splits.json").write_text(json.dumps(manifest))
    subprocess.run([sys.executable, "-m", "src.evaluate", "--predictions", str(tmp_path/"predictions.csv"),
                    "--training-targets", str(tmp_path/"training.csv"), "--manifest", str(tmp_path/"splits.json"),
                    "--output", str(tmp_path/"SIMULATED_REPORT")], check=True)
    hashes = json.loads((tmp_path/"SIMULATED_REPORT/input_hashes.json").read_text())
    assert set(hashes) == {"predictions", "training_targets", "manifest"}


def test_undefined_r2_and_zero_baseline_report(simulated, tmp_path):
    _, manifest, predictions, training = simulated
    predictions["actual_kwh"] = 1.0
    predictions["pred_kwh"] = 1.0
    training["actual_kwh"] = 1.0
    tables, decision = evaluate(predictions, manifest, training)
    assert tables["validation_summary"].R2_mean.isna().all()
    assert tables["validation_summary"].MAE_improvement_pct.isna().all()
    write_report(tables, decision, tmp_path/"SIMULATED_CONSTANT")
    assert (tmp_path/"SIMULATED_CONSTANT/comparison_R2.png").exists()
