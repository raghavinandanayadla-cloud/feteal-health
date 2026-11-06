"""Contract tests for the inference API, independent of trained artifacts."""

import json

import numpy as np
from fastapi.testclient import TestClient

from api import main
from api.monitoring_logger import log_event
from src.preprocess import EXPECTED_FEATURES


class ExampleModel:
    classes_ = np.array([1, 2, 3])

    def predict(self, features):
        assert tuple(features.columns) == EXPECTED_FEATURES
        return np.array([2])

    def predict_proba(self, features):
        return np.array([[0.1, 0.8, 0.1]])


def sample_features() -> dict[str, float]:
    return {name: 0.25 for name in EXPECTED_FEATURES}


def test_homepage_shows_project_dashboard():
    response = TestClient(main.app).get("/")

    assert response.status_code == 200
    assert "Fetal Health Prediction" in response.text
    assert "/docs" in response.text
    assert "Model evaluation" in response.text


def test_project_info_exposes_feature_contract():
    response = TestClient(main.app).get("/project-info")

    assert response.status_code == 200
    assert len(response.json()["features"]) == len(EXPECTED_FEATURES)
    assert response.json()["classes"]["3"] == "Pathological"


def test_predict_returns_class_and_probabilities(monkeypatch, tmp_path):
    log_path = tmp_path / "inference.jsonl"
    monkeypatch.setattr(main, "load_model", lambda: ExampleModel())
    monkeypatch.setattr(
        main,
        "log_event",
        lambda **event: log_event(**event, path=log_path),
    )

    response = TestClient(main.app).post(
        "/predict", json={"features": sample_features()}
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 2
    assert response.json()["label"] == "Suspect"
    assert response.json()["probabilities"] == {
        "Normal": 0.1,
        "Suspect": 0.8,
        "Pathological": 0.1,
    }
    event = json.loads(log_path.read_text(encoding="utf-8"))
    assert event["status"] == "ok"
    assert "features" not in event


def test_predict_rejects_missing_or_unexpected_features():
    client = TestClient(main.app)
    incomplete = sample_features()
    incomplete.pop(EXPECTED_FEATURES[0])
    extra = {**sample_features(), "unexpected": 1}

    assert client.post("/predict", json={"features": incomplete}).status_code == 422
    assert client.post("/predict", json={"features": extra}).status_code == 422


def test_predict_rejects_non_finite_feature_values():
    features = sample_features()
    features[EXPECTED_FEATURES[0]] = float("nan")

    response = TestClient(main.app).post(
        "/predict",
        content=json.dumps({"features": features}, allow_nan=True),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422


def test_health_reports_missing_model(monkeypatch):
    def unavailable_model():
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(main, "load_model", unavailable_model)

    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "unavailable", "model_loaded": False}