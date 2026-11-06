"""Failure-path tests for model loading and request handling."""

import json

from fastapi.testclient import TestClient

from api import main
from api.monitoring_logger import log_event
from src.preprocess import EXPECTED_FEATURES


def test_predict_returns_service_unavailable_when_model_load_fails(
    monkeypatch, tmp_path
):
    log_path = tmp_path / "inference.jsonl"

    def unavailable_model():
        raise FileNotFoundError("deployment artifact missing")

    monkeypatch.setattr(main, "load_model", unavailable_model)
    monkeypatch.setattr(
        main,
        "log_event",
        lambda **event: log_event(**event, path=log_path),
    )
    features = {name: 0.0 for name in EXPECTED_FEATURES}

    response = TestClient(main.app).post("/predict", json={"features": features})

    assert response.status_code == 503
    assert response.json()["detail"] == "Prediction service is temporarily unavailable."
    event = json.loads(log_path.read_text(encoding="utf-8"))
    assert event["status"] == "error"
    assert event["error_type"] == "FileNotFoundError"
    assert "features" not in event


def test_health_does_not_expose_model_loader_error(monkeypatch):
    def unavailable_model():
        raise RuntimeError("internal path detail")

    monkeypatch.setattr(main, "load_model", unavailable_model)

    response = TestClient(main.app).get("/health")

    assert response.json() == {"status": "unavailable", "model_loaded": False}
    assert "internal path detail" not in response.text