"""Request and response schemas for fetal-health inference."""

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.preprocess import EXPECTED_FEATURES


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        description="One observation with the 21 fetal cardiotocography features."
    )

    @field_validator("features")
    @classmethod
    def validate_feature_names_and_values(cls, features: dict[str, float]):
        missing = sorted(set(EXPECTED_FEATURES) - set(features))
        unexpected = sorted(set(features) - set(EXPECTED_FEATURES))
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing features: {missing}")
            if unexpected:
                details.append(f"unexpected features: {unexpected}")
            raise ValueError("Invalid feature set; " + "; ".join(details))
        invalid_values = [
            name for name, value in features.items() if not math.isfinite(value)
        ]
        if invalid_values:
            raise ValueError(f"Feature values must be finite: {invalid_values}")
        return features


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool