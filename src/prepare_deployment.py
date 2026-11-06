"""Package the selected local model with a verifiable metadata manifest."""

import hashlib
import json
import joblib
import shutil
from datetime import datetime, timezone

from src.evaluate import calculate_metrics
from src.preprocess import EXPECTED_FEATURES, PROJECT_ROOT
from src.train import MODEL_PATH, prepare_data

DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
DEPLOYED_MODEL_PATH = DEPLOYMENT_DIR / "model.pkl"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "production_model_report.json"


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_deployment() -> dict:
    """Copy the trained pipeline and write its feature and integrity metadata."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}; run Lab 3/4 first."
        )

    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_PATH, DEPLOYED_MODEL_PATH)
    model = joblib.load(DEPLOYED_MODEL_PATH)
    _, test_features, _, test_target = prepare_data()
    evaluation = calculate_metrics(test_target, model.predict(test_features))
    classifier = model.named_steps["classifier"]
    model_names = {
        "LogisticRegression": "Logistic Regression",
        "DecisionTreeClassifier": "Decision Tree",
        "RandomForestClassifier": "Random Forest",
    }
    metadata = {
        "model_file": DEPLOYED_MODEL_PATH.name,
        "model_name": model_names.get(
            type(classifier).__name__, type(classifier).__name__
        ),
        "sha256": _sha256(DEPLOYED_MODEL_PATH),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "features": list(EXPECTED_FEATURES),
        "classes": {"1": "Normal", "2": "Suspect", "3": "Pathological"},
        "evaluation": evaluation,
        "evaluation_split": "stratified held-out test partition",
        "random_seed": 42,
        "intended_use": "Educational demonstration only; not a clinical decision tool.",
    }
    serialized = json.dumps(metadata, indent=2) + "\n"
    METADATA_PATH.write_text(serialized, encoding="utf-8")
    REPORT_PATH.write_text(serialized, encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = prepare_deployment()
    print(f"Deployment model: {DEPLOYED_MODEL_PATH}")
    print(f"SHA-256: {result['sha256']}")