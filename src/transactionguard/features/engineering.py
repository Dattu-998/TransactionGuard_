"""Feature definitions shared by training and serving.

Every model version is a single scikit-learn estimator that accepts the raw
``Time, V1..V28, Amount`` frame, so preprocessing can never drift between the
training script and the API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time", *PCA_COLUMNS, "Amount"]
NIGHT_HOURS = (0, 6)


def feature_label(name: str) -> str:
    """Human-readable label for a raw input column."""
    if name == "Amount":
        return "Transaction amount"
    if name == "Time":
        return "Time of day"
    return f"Behavioral component {name}"


def hour_of_day(time_seconds: pd.Series | np.ndarray) -> np.ndarray:
    return (np.asarray(time_seconds, dtype=float) / 3600.0) % 24.0


class TransactionFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds behavioural features on top of the anonymised PCA components.

    * cyclical hour-of-day encoding and a night-time flag (fraud peaks overnight)
    * log amount, a micro-charge flag (card testing) and amount relative to the
      training median
    * the L2 norm of the PCA vector, a cheap "how unusual is this point" signal
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.amount_median_ = float(np.median(X["Amount"])) or 1.0
        self.feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        hours = hour_of_day(X["Time"])
        amount = X["Amount"].to_numpy(dtype=float)
        components = X[PCA_COLUMNS].to_numpy(dtype=float)
        out = pd.DataFrame(components, columns=PCA_COLUMNS, index=X.index)
        out["log_amount"] = np.log1p(np.clip(amount, 0, None))
        out["amount_vs_median"] = np.log1p(np.clip(amount, 0, None)) - np.log1p(self.amount_median_)
        out["micro_charge"] = (amount < 5.0).astype(float)
        out["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        out["is_night"] = ((hours >= NIGHT_HOURS[0]) & (hours < NIGHT_HOURS[1])).astype(float)
        out["component_norm"] = np.sqrt((components**2).sum(axis=1))
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            [*PCA_COLUMNS, "log_amount", "amount_vs_median", "micro_charge",
             "hour_sin", "hour_cos", "is_night", "component_norm"],
            dtype=object,
        )


def build_baseline_preprocessor() -> ColumnTransformer:
    """The original v1 preprocessing: scale Time and Amount, pass PCA components through."""
    return ColumnTransformer(
        transformers=[
            ("scaled", StandardScaler(), ["Time", "Amount"]),
            ("components", "passthrough", PCA_COLUMNS),
        ],
        remainder="drop",
    )


def feature_statistics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-feature summary used for UI ranges, explanation baselines and input checks."""
    stats = {}
    for column in FEATURE_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        q = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
        stats[column] = {
            "label": feature_label(column),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p01": float(q[0]),
            "p05": float(q[1]),
            "p25": float(q[2]),
            "median": float(q[3]),
            "p75": float(q[4]),
            "p95": float(q[5]),
            "p99": float(q[6]),
        }
    return stats
