"""Lab 10: retrain, package, and report the local model lifecycle."""

import json

from src.preprocess import PROJECT_ROOT
from src.prepare_deployment import prepare_deployment
from src.train_mlflow import run_mlflow

REPORT_PATH = PROJECT_ROOT / "reports" / "lab10_lifecycle_report.json"


def run() -> dict:
    training = run_mlflow()
    deployment = prepare_deployment()
    report = {
        "status": "READY",
        "selected_model": training["best_model_name"],
        "metrics": training["metrics"],
        "deployment_model": deployment["model_file"],
        "model_sha256": deployment["sha256"],
        "intended_use": deployment["intended_use"],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Lifecycle report: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run()