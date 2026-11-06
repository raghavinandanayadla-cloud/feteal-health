"""Lab 12: compare current fetal-health data with a reference dataset."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.monitoring import analyze_monitoring_data, write_monitoring_report
from src.preprocess import EXPECTED_FEATURES, PROJECT_ROOT

REFERENCE_PATH = PROJECT_ROOT / "data" / "raw" / "fetal_health.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "lab12_monitoring_report.json"
DEPLOYED_MODEL_PATH = PROJECT_ROOT / "deployment" / "model.pkl"


def run(reference_path=REFERENCE_PATH, current_path=REFERENCE_PATH) -> dict:
    reference_path = Path(reference_path)
    current_path = Path(current_path)
    reference = pd.read_csv(reference_path).drop_duplicates().reset_index(drop=True)
    current = pd.read_csv(current_path).drop_duplicates().reset_index(drop=True)
    same_dataset = reference_path.resolve() == current_path.resolve()
    predictions = None
    model_status = "not_available"
    if DEPLOYED_MODEL_PATH.is_file() and same_dataset:
        model_status = "available_no_independent_current_data"
    elif DEPLOYED_MODEL_PATH.is_file():
        model = joblib.load(DEPLOYED_MODEL_PATH)
        predictions = model.predict(current.loc[:, list(EXPECTED_FEATURES)])
        model_status = "evaluated"

    report = analyze_monitoring_data(reference, current, predictions)
    report["model_status"] = model_status
    report["performance_metrics_note"] = (
        "Skipped because current data is the reference dataset."
        if same_dataset
        else "Labeled metrics are descriptive; confirm the current data is independent."
    )
    report["reference_path"] = str(reference_path)
    report["current_path"] = str(current_path)
    write_monitoring_report(report, REPORT_PATH)
    print(f"Monitoring status: {report['status']}")
    print(f"Drifted feature count: {len(report['drifted_features'])}")
    print(f"Monitoring report: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=str, default=str(REFERENCE_PATH))
    parser.add_argument("--current", type=str, default=str(REFERENCE_PATH))
    arguments = parser.parse_args()
    run(arguments.reference, arguments.current)