import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        self.model = None
        self.feature_engineer = None
        self.metadata = {}
        self.schema = {}
        self.is_loaded = False

    def load(self):
        try:
            model_path = self.artifact_dir / "model.joblib"
            fe_path = self.artifact_dir / "feature_engineer.joblib"
            meta_path = self.artifact_dir / "metadata.json"
            schema_path = self.artifact_dir / "schema.json"

            if not all(p.exists() for p in [model_path, fe_path, meta_path, schema_path]):
                return False

            self.model = joblib.load(model_path)
            self.feature_engineer = joblib.load(fe_path)
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)

            self.is_loaded = True
            return True
        except (json.JSONDecodeError, FileNotFoundError, OSError, TypeError) as e:
            logger.error(f"Failed to load model artifacts: {e}")
            return False

    def get_info(self) -> dict:
        return {
            "model_name": self.metadata.get("model_name", "Unknown"),
            "version": self.metadata.get("version", "v1.0"),
            "threshold": self.metadata.get("threshold", 0.5),
            "features": self.schema
        }

    def validate_features(self, df: pd.DataFrame) -> tuple[bool, str | None]:
        missing = [f for f in self.schema if f not in df.columns]
        if missing:
            return False, f"Missing required features: {missing}"

        unexpected = [f for f in df.columns if f not in self.schema]
        if unexpected:
            return False, f"Unexpected features found: {unexpected}"

        try:
            for col in self.schema:
                if df[col].isnull().any():
                    return False, f"Missing values (NaN) found in column {col}"
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any() or not pd.api.types.is_numeric_dtype(df[col]):
                    return False, f"Non-numeric values found in column {col}"
                if np.isinf(df[col]).any():
                    return False, f"Infinite values found in column {col}"
        except (ValueError, TypeError) as e:
            return False, f"Data validation error: {e}"

        return True, None

    def predict(self, df: pd.DataFrame) -> list[dict]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        df = df[list(self.schema.keys())]

        X = self.feature_engineer.transform(df)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
        else:
            probs = self.model.predict(X)
        threshold = self.metadata.get("threshold", 0.5)
        model_version = self.metadata.get("version", "v1.0")

        predictions = []
        for p in probs:
            is_fraud = bool(p >= threshold)
            predictions.append({
                "probability": float(p),
                "is_fraud": is_fraud,
                "threshold": float(threshold),
                "model_version": str(model_version),
                "explanation": f"Probability {p:.4f} " + (">=" if is_fraud else "<") + f" threshold {threshold:.4f}"
            })
        return predictions
