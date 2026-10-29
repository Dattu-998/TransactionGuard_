"""File-based model registry.

Layout under the artifact directory::

    registry.json            champion pointer + per-version summaries
    feature_stats.json       training feature statistics (UI ranges, baselines)
    drift_reference.json     training distributions for PSI monitoring
    samples/demo_transactions.csv
    models/<version>/model.joblib
    models/<version>/card.json   full model card (metrics, curves, importance)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from transactionguard.features.engineering import FEATURE_COLUMNS
from transactionguard.scoring import (
    DecisionPolicy,
    ScoringError,
    explain,
    positive_probabilities,
    risk_level,
    validate_frame,
)

REGISTRY_FILE = "registry.json"


@dataclass
class LoadedModel:
    version: str
    card: dict
    estimator: object

    @property
    def policy(self) -> DecisionPolicy:
        thresholds = self.card["thresholds"]
        return DecisionPolicy(thresholds["review"], thresholds["block"])


class ModelRegistry:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.index: dict = {}
        self.models: dict[str, LoadedModel] = {}
        self.feature_stats: dict = {}
        self.drift_reference: dict = {}
        self.samples: pd.DataFrame | None = None

    # ------------------------------------------------------------------ loading
    def load(self) -> ModelRegistry:
        index_path = self.artifact_dir / REGISTRY_FILE
        if not index_path.exists():
            raise ScoringError(f"No model registry found at {index_path}.")
        try:
            self.index = json.loads(index_path.read_text(encoding="utf-8"))
            for entry in self.index["versions"]:
                folder = self.artifact_dir / "models" / entry["version"]
                card = json.loads((folder / "card.json").read_text(encoding="utf-8"))
                estimator = joblib.load(folder / "model.joblib")
                self.models[entry["version"]] = LoadedModel(entry["version"], card, estimator)
            self.feature_stats = self._read_json("feature_stats.json")
            self.drift_reference = self._read_json("drift_reference.json")
            samples_path = self.artifact_dir / "samples" / "demo_transactions.csv"
            if samples_path.exists():
                self.samples = pd.read_csv(samples_path)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise ScoringError(f"Unable to load the model registry: {exc}") from exc
        if self.index.get("champion") not in self.models:
            raise ScoringError("The registry champion is not among the loaded models.")
        return self

    def _read_json(self, name: str) -> dict:
        path = self.artifact_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    # --------------------------------------------------------------- accessors
    @property
    def champion(self) -> str:
        return self.index["champion"]

    @property
    def versions(self) -> list[str]:
        return list(self.models)

    def get(self, version: str | None = None) -> LoadedModel:
        key = version or self.champion
        if key not in self.models:
            raise ScoringError(f"Unknown model version '{key}'. Available: {', '.join(self.models)}.")
        return self.models[key]

    @property
    def baseline(self) -> dict[str, float]:
        return {name: self.feature_stats[name]["median"] for name in FEATURE_COLUMNS}

    # ----------------------------------------------------------------- scoring
    def score(self, frame: pd.DataFrame, version: str | None = None, explain_top: int = 6,
              explain_rows: str = "flagged") -> tuple[pd.DataFrame, list[list[dict]], float]:
        """Score rows; returns (results, explanations per row, elapsed ms).

        ``explain_rows`` is ``"all"``, ``"flagged"`` (review/block only) or ``"none"``.
        """
        model = self.get(version)
        features = validate_frame(frame)
        started = time.perf_counter()
        try:
            probabilities = positive_probabilities(model.estimator, features)
        except (ValueError, TypeError, KeyError) as exc:
            raise ScoringError("The transaction data is incompatible with this model.") from exc
        policy = model.policy
        scores = policy.risk_scores(probabilities)
        decisions = policy.decisions(probabilities)
        results = pd.DataFrame({
            "fraud_probability": probabilities,
            "risk_score": np.round(scores, 1),
            "risk_level": [risk_level(s) for s in scores],
            "decision": decisions,
        }, index=frame.index)

        explanations: list[list[dict]] = [[] for _ in range(len(frame))]
        if explain_rows != "none" and self.feature_stats:
            mask = np.ones(len(frame), bool) if explain_rows == "all" else decisions != "approve"
            if mask.any():
                subset = features.loc[mask]
                attributions = explain(model.estimator, subset, self.baseline, explain_top,
                                       probabilities=probabilities[mask])
                for position, reasons in zip(np.flatnonzero(mask), attributions, strict=True):
                    explanations[position] = reasons
        elapsed_ms = (time.perf_counter() - started) * 1000
        return results, explanations, elapsed_ms
