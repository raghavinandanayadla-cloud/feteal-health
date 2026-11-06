"""Lab 13: smoke-test the packaged model through the HTTP API."""

import json

import pandas as pd
from fastapi.testclient import TestClient

from api import main
from src.preprocess import EXPECTED_FEATURES, PROJECT_ROOT, RAW_DATA_PATH

REPORT_PATH = PROJECT_ROOT / "artifacts" / "api_test_report.json"


def run() -> dict:
    main.load_model.cache_clear()
    client = TestClient(main.app)
    health_response = client.get("/health")
    sample = pd.read_csv(RAW_DATA_PATH).loc[0, list(EXPECTED_FEATURES)]
    prediction_response = client.post(
        "/predict", json={"features": sample.to_dict()}
    )
    report = {
        "status": "PASS"
        if health_response.status_code == 200
        and health_response.json()["model_loaded"]
        and prediction_response.status_code == 200
        else "FAIL",
        "health_status_code": health_response.status_code,
        "prediction_status_code": prediction_response.status_code,
        "prediction": prediction_response.json().get("prediction"),
        "label": prediction_response.json().get("label"),
        "clinical_use": False,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"API end-to-end status: {report['status']}")
    print(f"API report: {REPORT_PATH}")
    if report["status"] != "PASS":
        raise RuntimeError(f"API end-to-end check failed: {report}")
    return report


if __name__ == "__main__":
    run()