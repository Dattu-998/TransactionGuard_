"""Turns model probabilities into analyst-facing decisions.

A decision policy has two cut-offs learned on validation data:

* ``review`` – the cost-optimal threshold; at or above it a transaction goes to
  an analyst queue.
* ``block`` – the lowest threshold whose validation precision reaches the target;
  at or above it the transaction is declined automatically.

The 0–100 risk score is a monotone, piecewise log-scale mapping that pins the
review threshold to 50 and the block threshold to 80, so scores read the same
way across model versions even though their raw probabilities differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transactionguard.features.engineering import FEATURE_COLUMNS, feature_label

PROBABILITY_FLOOR = 1e-6
REVIEW_SCORE = 50.0
BLOCK_SCORE = 80.0


class ScoringError(ValueError):
    """Raised when input or model artifacts cannot be used for scoring."""


@dataclass(frozen=True)
class DecisionPolicy:
    review_threshold: float
    block_threshold: float

    def risk_scores(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probabilities, dtype=float), PROBABILITY_FLOOR, 1.0)
        lp = np.log10(p)
        lo, rv = np.log10(PROBABILITY_FLOOR), np.log10(max(self.review_threshold, 1.1e-6))
        bl = np.log10(min(max(self.block_threshold, self.review_threshold * 1.0001), 0.9999))
        anchors_x = [lo, rv, bl, 0.0]
        anchors_y = [0.0, REVIEW_SCORE, BLOCK_SCORE, 100.0]
        return np.interp(lp, anchors_x, anchors_y)

    def decisions(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        return np.where(p >= self.block_threshold, "block",
                        np.where(p >= self.review_threshold, "review", "approve"))


def risk_level(score: float) -> str:
    if score >= BLOCK_SCORE:
        return "critical"
    if score >= REVIEW_SCORE:
        return "high"
    if score >= 30:
        return "elevated"
    return "low"


def validate_frame(frame: pd.DataFrame, columns: list[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """Return numeric model inputs or raise with every problem found."""
    if frame.empty:
        raise ScoringError("The request contains no transaction rows.")
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ScoringError(f"Missing required model feature(s): {', '.join(missing)}.")
    features = frame.loc[:, columns].copy()
    errors: list[str] = []
    for column in columns:
        original = features[column]
        converted = pd.to_numeric(original, errors="coerce")
        if original.isna().any():
            errors.append(f"{column} contains missing values.")
        elif converted.isna().any():
            errors.append(f"{column} contains non-numeric values.")
        elif np.isinf(converted).any():
            errors.append(f"{column} contains infinite values.")
        features[column] = converted.astype(float)
    if errors:
        raise ScoringError(" ".join(errors))
    return features


def positive_probabilities(model, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features))
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        raise ScoringError("The model has no fraud (class 1) probability column.")
    return probabilities[:, classes.index(1)].astype(float)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    return np.log(p / (1 - p))


def explain(
    model,
    features: pd.DataFrame,
    baseline: dict[str, float],
    top_k: int = 6,
    probabilities: np.ndarray | None = None,
) -> list[list[dict]]:
    """Baseline-substitution attributions in log-odds space.

    For each input, replace it with the typical (median) training value and measure
    how much the fraud log-odds fall. Positive contributions push towards fraud.
    The method is model-agnostic, so every registry version explains itself the
    same way.
    """
    if probabilities is None:
        probabilities = positive_probabilities(model, features)
    base_logit = _logit(np.asarray(probabilities, dtype=float))
    n = len(features)
    columns = list(features.columns)
    values = np.tile(features.to_numpy(dtype=float), (len(columns), 1))
    for i, column in enumerate(columns):
        values[i * n:(i + 1) * n, i] = baseline[column]
    stacked = pd.DataFrame(values, columns=columns)
    substituted = _logit(positive_probabilities(model, stacked)).reshape(len(columns), n)
    contributions = base_logit[None, :] - substituted  # (features, rows)

    explanations = []
    for row in range(n):
        order = np.argsort(-np.abs(contributions[:, row]))[:top_k]
        explanations.append([
            {
                "feature": columns[j],
                "label": feature_label(columns[j]),
                "value": float(features.iloc[row, j]),
                "baseline": float(baseline[columns[j]]),
                "contribution": float(contributions[j, row]),
                "direction": "increases_risk" if contributions[j, row] > 0 else "decreases_risk",
            }
            for j in order
            if abs(contributions[j, row]) > 1e-4
        ])
    return explanations
