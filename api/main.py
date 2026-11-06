"""FastAPI application for local fetal-health model inference."""

import json
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from api.model_loader import load_model
from api.monitoring_logger import log_event
from api.preprocessing import features_to_frame
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.preprocess import EXPECTED_FEATURES, PROJECT_ROOT

CLASS_LABELS = {1: "Normal", 2: "Suspect", 3: "Pathological"}
INDEX_PATH = Path(__file__).with_name("index.html")
MODEL_METADATA_PATH = PROJECT_ROOT / "deployment" / "model_metadata.json"
app = FastAPI(
    title="Fetal Health Prediction API",
    description="Educational inference service; not for clinical decision-making.",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
def request_validation_error_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    details = [
        {"loc": item.get("loc"), "msg": item.get("msg"), "type": item.get("type")}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/", include_in_schema=False)
def homepage() -> FileResponse:
    return FileResponse(INDEX_PATH, media_type="text/html")


@app.get("/project-info", include_in_schema=False)
def project_info() -> dict:
    metadata = {}
    if MODEL_METADATA_PATH.is_file():
        try:
            metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    classifier_name = metadata.get("model_name", "Fetal health classifier")
    return {
        "project_name": "Fetal Health Prediction",
        "model_name": classifier_name,
        "model_file": metadata.get("model_file"),
        "features": list(EXPECTED_FEATURES),
        "classes": {str(class_id): name for class_id, name in CLASS_LABELS.items()},
        "evaluation": metadata.get("evaluation", {}),
        "intended_use": "Educational demonstration only; not a clinical decision tool.",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        load_model()
    except Exception:
        return HealthResponse(status="unavailable", model_loaded=False)
    return HealthResponse(status="ok", model_loaded=True)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    request_id = str(uuid4())
    started_at = time.perf_counter()
    try:
        model_input = features_to_frame(request.features)
        model = load_model()
        prediction = int(model.predict(model_input)[0])
        if prediction not in CLASS_LABELS:
            raise ValueError(f"Model returned an unsupported class: {prediction}")
        probabilities = model.predict_proba(model_input)[0]
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = model.named_steps["classifier"].classes_
        result_probabilities = {
            CLASS_LABELS[int(class_id)]: float(probability)
            for class_id, probability in zip(classes, probabilities, strict=True)
        }
        response = PredictionResponse(
            prediction=prediction,
            label=CLASS_LABELS[prediction],
            probabilities=result_probabilities,
        )
        log_event(
            request_id=request_id,
            status="ok",
            latency_ms=(time.perf_counter() - started_at) * 1000,
            prediction=prediction,
        )
        return response
    except Exception as error:
        log_event(
            request_id=request_id,
            status="error",
            latency_ms=(time.perf_counter() - started_at) * 1000,
            error_type=type(error).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Prediction service is temporarily unavailable.",
        ) from error