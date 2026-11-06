"""Load the packaged sklearn pipeline used by the inference API."""

import os
from functools import lru_cache
from pathlib import Path

import joblib

from src.preprocess import EXPECTED_FEATURES, PROJECT_ROOT

DEFAULT_MODEL_PATH = PROJECT_ROOT / "deployment" / "model.pkl"


@lru_cache(maxsize=2)
def load_model(model_path: str | None = None):
    """Load and validate a fitted model pipeline from disk."""
    configured_path = model_path or os.environ.get("FETAL_MODEL_PATH")
    path = Path(configured_path) if configured_path else DEFAULT_MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run Lab 7 deployment preparation first."
        )

    model = joblib.load(path)
    if not callable(getattr(model, "predict", None)):
        raise TypeError("Loaded model does not provide a predict method.")
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("Loaded model does not provide a predict_proba method.")
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and tuple(model_features) != EXPECTED_FEATURES:
        raise ValueError("Model feature names do not match the API feature contract.")
    classifier = getattr(model, "named_steps", {}).get("classifier")
    if classifier is not None and hasattr(classifier, "n_jobs"):
        classifier.set_params(n_jobs=1)
    return model