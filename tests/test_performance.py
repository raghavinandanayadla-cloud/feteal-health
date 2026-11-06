"""Smoke-test repeated API predictions without a brittle latency gate."""

import time

import numpy as np
from fastapi.testclient import TestClient

from api import main
from src.preprocess import EXPECTED_FEATURES


class FastExampleModel:
    classes_ = np.array([1, 2, 3])

    def predict(self, features):
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.8, 0.1, 0.1]])


def test_repeated_predictions_return_valid_responses(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda: FastExampleModel())
    monkeypatch.setattr(main, "log_event", lambda **event: None)
    payload = {"features": {name: 0.5 for name in EXPECTED_FEATURES}}
    durations = []

    with TestClient(main.app) as client:
        for _ in range(5):
            started_at = time.perf_counter()
            response = client.post("/predict", json=payload)
            durations.append(time.perf_counter() - started_at)
            assert response.status_code == 200
            assert response.json()["prediction"] == 1

    assert len(durations) == 5
    assert all(duration > 0 for duration in durations)