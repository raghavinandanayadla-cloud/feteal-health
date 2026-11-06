"""Convert validated API inputs into the model's training feature order."""

import math

import pandas as pd

from src.preprocess import EXPECTED_FEATURES


def features_to_frame(features: dict[str, float]) -> pd.DataFrame:
    """Validate feature keys and values, then create a one-row model input."""
    missing = sorted(set(EXPECTED_FEATURES) - set(features))
    unexpected = sorted(set(features) - set(EXPECTED_FEATURES))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing features: {missing}")
        if unexpected:
            details.append(f"unexpected features: {unexpected}")
        raise ValueError("Invalid feature set; " + "; ".join(details))

    values = {name: float(features[name]) for name in EXPECTED_FEATURES}
    invalid = [name for name, value in values.items() if not math.isfinite(value)]
    if invalid:
        raise ValueError(f"Feature values must be finite: {invalid}")
    return pd.DataFrame([values], columns=list(EXPECTED_FEATURES))