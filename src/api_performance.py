"""Measure in-process inference endpoint latency and write a JSON report."""

import json
import math
import statistics
import time

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from src.preprocess import EXPECTED_FEATURES, PROJECT_ROOT, RAW_DATA_PATH

REPORT_PATH = PROJECT_ROOT / "artifacts" / "api_performance_report.json"


def run_api_performance(sample_count: int = 30) -> dict:
    """Measure request latency for dataset rows through the in-process API."""
    if sample_count < 1:
        raise ValueError("sample_count must be at least one")
    sample_data = pd.read_csv(RAW_DATA_PATH).loc[:, list(EXPECTED_FEATURES)]
    samples = sample_data.head(sample_count)
    latencies_ms = []
    with TestClient(app) as client:
        health_status = client.get("/health").status_code
        for _, row in samples.iterrows():
            started_at = time.perf_counter()
            response = client.post("/predict", json={"features": row.to_dict()})
            latencies_ms.append((time.perf_counter() - started_at) * 1000)
            if response.status_code != 200:
                raise RuntimeError(
                    f"API performance request failed with status {response.status_code}."
                )

    ordered_latencies = sorted(latencies_ms)
    report = {
        "status": "PASS" if health_status == 200 and len(latencies_ms) == len(samples) else "FAIL",
        "measurement": "in-process FastAPI TestClient; excludes network and container overhead",
        "requests": len(latencies_ms),
        "health_status_code": health_status,
        "latency_ms": {
            "mean": statistics.mean(latencies_ms),
            "p50": statistics.median(latencies_ms),
            "p95": ordered_latencies[math.ceil(0.95 * len(ordered_latencies)) - 1],
            "max": max(latencies_ms),
        },
        "production_sla_configured": False,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"API performance report: {REPORT_PATH}")
    return report