"""Safe, reusable local inference helpers for the TransactionGuard UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class InferenceError(ValueError):
    """Raised when input or model artifacts cannot be used for inference."""


@dataclass(frozen=True)
class ModelDetails:
    name: str
    threshold: float
    version: str
    feature_names: list[str]


def risk_category(probability: float) -> str:
    """Presentation-only risk band; it does not alter the decision threshold."""
    if probability < 0.20:
        return "Low"
    if probability < 0.60:
        return "Medium"
    return "High"


class InferenceService:
    """Loads the existing model artifacts and applies their fitted transformer."""

    def __init__(self, artifact_dir: str | Path = "artifacts"):
        self.artifact_dir = Path(artifact_dir)
        self.model = None
        self.feature_engineer = None
        self.schema: dict[str, str] = {}
        self.metadata: dict = {}

    def load(self) -> InferenceService:
        paths = {
            "model": self.artifact_dir / "model.joblib",
            "feature engineer": self.artifact_dir / "feature_engineer.joblib",
            "metadata": self.artifact_dir / "metadata.json",
            "schema": self.artifact_dir / "schema.json",
        }
        absent = [name for name, path in paths.items() if not path.exists()]
        if absent:
            raise InferenceError(f"Missing model artifact(s): {', '.join(absent)}.")
        try:
            self.model = joblib.load(paths["model"])
            self.feature_engineer = joblib.load(paths["feature engineer"])
            self.metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            self.schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise InferenceError("Unable to load model artifacts.") from exc
        if not self.schema:
            raise InferenceError("The saved feature schema is empty.")
        return self

    @property
    def details(self) -> ModelDetails:
        return ModelDetails(
            name=str(self.metadata.get("model_name", "Unknown model")),
            threshold=float(self.metadata.get("threshold", 0.5)),
            version=str(self.metadata.get("version", "unknown")),
            feature_names=list(self.schema),
        )

    def validate(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Return model-only numeric columns plus human-readable validation errors."""
        if frame.empty:
            return frame.copy(), ["The uploaded file has no transaction rows."]
        missing = [name for name in self.schema if name not in frame.columns]
        if missing:
            return frame.copy(), [f"Missing required model feature(s): {', '.join(missing)}."]
        features = frame.loc[:, list(self.schema)].copy()
        errors: list[str] = []
        for column in features:
            original = features[column]
            converted = pd.to_numeric(original, errors="coerce")
            invalid = converted.isna() & original.notna()
            if original.isna().any():
                errors.append(f"{column} contains missing values.")
            if invalid.any():
                errors.append(f"{column} contains non-numeric values.")
            if np.isinf(converted).any():
                errors.append(f"{column} contains infinite values.")
            features[column] = converted
        return features, errors

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.feature_engineer is None:
            raise InferenceError("Model artifacts have not been loaded.")
        features, errors = self.validate(frame)
        if errors:
            raise InferenceError(" ".join(errors))
        try:
            transformed = self.feature_engineer.transform(features)
            probabilities = self._positive_probabilities(transformed)
        except (ValueError, TypeError, KeyError) as exc:
            raise InferenceError("The transaction data is incompatible with this model.") from exc
        threshold = self.details.threshold
        fraud = probabilities >= threshold
        return pd.DataFrame(
            {
                "Fraud Probability": probabilities,
                "Predicted Class": np.where(fraud, "Potential Fraud", "Likely Legitimate"),
                "Risk Category": [risk_category(float(p)) for p in probabilities],
                "Prediction Status": np.where(fraud, "Flagged for review", "No fraud flag"),
            },
            index=frame.index,
        )

    def _positive_probabilities(self, transformed: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict(transformed), dtype=float)
        probabilities = np.asarray(self.model.predict_proba(transformed))
        classes = list(getattr(self.model, "classes_", [0, 1]))
        if 1 not in classes:
            raise InferenceError("The model has no fraud (class 1) probability column.")
        return probabilities[:, classes.index(1)].astype(float)
