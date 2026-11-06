"""Lightweight feature-drift and optional labeled-performance monitoring."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.preprocess import EXPECTED_FEATURES, TARGET_COLUMN

PSI_ALERT_THRESHOLD = 0.2


def _population_stability_index(reference: np.ndarray, current: np.ndarray) -> float:
    quantiles = np.quantile(reference, np.linspace(0, 1, 11))
    inner_edges = np.unique(quantiles[1:-1])
    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    current_counts = np.histogram(current, bins=edges)[0].astype(float)
    reference_share = (reference_counts + 1e-6) / (
        len(reference) + 1e-6 * len(reference_counts)
    )
    current_share = (current_counts + 1e-6) / (
        len(current) + 1e-6 * len(current_counts)
    )
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def analyze_monitoring_data(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    predictions: np.ndarray | None = None,
) -> dict:
    """Compare feature distributions and score predictions when labels exist."""
    reference_missing = sorted(set(EXPECTED_FEATURES) - set(reference.columns))
    missing = sorted(set(EXPECTED_FEATURES) - set(current.columns))
    if reference_missing:
        raise ValueError(f"Reference data is missing required features: {reference_missing}")
    if missing:
        raise ValueError(f"Current data is missing required features: {missing}")
    if reference.empty or current.empty:
        raise ValueError("Reference and current datasets must both contain rows.")

    feature_drift = {}
    for feature in EXPECTED_FEATURES:
        baseline_values = (
            pd.to_numeric(reference[feature], errors="coerce").dropna().to_numpy()
        )
        current_values = (
            pd.to_numeric(current[feature], errors="coerce").dropna().to_numpy()
        )
        if not len(baseline_values) or not len(current_values):
            raise ValueError(f"Feature '{feature}' has no usable numeric values.")
        if not np.isfinite(baseline_values).all() or not np.isfinite(current_values).all():
            raise ValueError(f"Feature '{feature}' contains non-finite values.")
        feature_drift[feature] = _population_stability_index(
            baseline_values, current_values
        )

    report = {
        "status": "DRIFT_ALERT"
        if any(value >= PSI_ALERT_THRESHOLD for value in feature_drift.values())
        else "PASS",
        "method": "population stability index using reference decile bins",
        "threshold": PSI_ALERT_THRESHOLD,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "feature_psi": feature_drift,
        "drifted_features": sorted(
            name
            for name, value in feature_drift.items()
            if value >= PSI_ALERT_THRESHOLD
        ),
        "clinical_use": False,
    }
    if TARGET_COLUMN in current.columns and predictions is not None:
        target = current[TARGET_COLUMN].to_numpy()
        if len(target) != len(predictions):
            raise ValueError("Prediction count must match the labeled current data.")
        report["labeled_metrics"] = {
            "accuracy": float(accuracy_score(target, predictions)),
            "macro_f1": float(f1_score(target, predictions, average="macro")),
        }
    return report


def write_monitoring_report(report: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")