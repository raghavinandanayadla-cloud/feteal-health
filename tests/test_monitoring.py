"""Tests for feature drift and optional labeled evaluation."""

import pandas as pd
import pytest

from pipelines import run_lab12_monitoring
from src.monitoring import analyze_monitoring_data
from src.preprocess import EXPECTED_FEATURES, TARGET_COLUMN


def monitoring_frame(rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {name: [float(index % 10) for index in range(rows)] for name in EXPECTED_FEATURES}
    )


def test_identical_distributions_do_not_raise_drift_alert():
    reference = monitoring_frame()

    report = analyze_monitoring_data(reference, reference.copy())

    assert report["status"] == "PASS"
    assert report["drifted_features"] == []


def test_shifted_feature_is_reported_as_drift():
    reference = monitoring_frame()
    current = reference.copy()
    current[EXPECTED_FEATURES[0]] += 100

    report = analyze_monitoring_data(reference, current)

    assert report["status"] == "DRIFT_ALERT"
    assert EXPECTED_FEATURES[0] in report["drifted_features"]


def test_monitoring_rejects_missing_features():
    reference = monitoring_frame()
    current = reference.drop(columns=[EXPECTED_FEATURES[0]])

    with pytest.raises(ValueError, match="missing required features"):
        analyze_monitoring_data(reference, current)


def test_default_monitoring_does_not_report_in_sample_metrics(monkeypatch, tmp_path):
    dataset_path = tmp_path / "reference.csv"
    report_path = tmp_path / "monitoring.json"
    model_path = tmp_path / "model.pkl"
    data = monitoring_frame()
    data[TARGET_COLUMN] = [1] * len(data)
    data.to_csv(dataset_path, index=False)
    model_path.write_bytes(b"not loaded for self-comparison")
    monkeypatch.setattr(run_lab12_monitoring, "DEPLOYED_MODEL_PATH", model_path)
    monkeypatch.setattr(run_lab12_monitoring, "REPORT_PATH", report_path)

    report = run_lab12_monitoring.run(dataset_path, dataset_path)

    assert report["status"] == "PASS"
    assert report["model_status"] == "available_no_independent_current_data"
    assert "labeled_metrics" not in report