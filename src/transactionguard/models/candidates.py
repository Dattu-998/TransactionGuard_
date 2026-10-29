"""Model versions trained into the registry.

Each candidate is a complete estimator over the raw input frame. Versions are
ordered by sophistication; the registry promotes whichever scores best on the
validation set, so the champion is earned rather than assumed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from transactionguard.config import config
from transactionguard.features.engineering import (
    TransactionFeatureBuilder,
    build_baseline_preprocessor,
)


@dataclass(frozen=True)
class Candidate:
    version: str
    name: str
    family: str
    summary: str
    highlights: list[str]
    params: dict = field(default_factory=dict)
    build: Callable[[float], object] = field(default=lambda _: None, repr=False)


def _v1(_: float):
    return Pipeline([
        ("preprocess", build_baseline_preprocessor()),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=config.RANDOM_STATE)),
    ])


V2_PARAMS = {"n_estimators": 250, "max_depth": 14, "min_samples_leaf": 4,
             "max_features": "sqrt", "class_weight": "balanced_subsample"}


def _v2(_: float):
    return Pipeline([
        ("features", TransactionFeatureBuilder()),
        ("classifier", RandomForestClassifier(
            **V2_PARAMS, n_jobs=-1, random_state=config.RANDOM_STATE)),
    ])


V3_PARAMS = {"n_estimators": 700, "learning_rate": 0.03, "num_leaves": 31,
             "min_child_samples": 25, "subsample": 0.85, "subsample_freq": 1,
             "colsample_bytree": 0.75, "reg_lambda": 2.0}


def _v3(imbalance_ratio: float):
    booster = lgb.LGBMClassifier(
        **V3_PARAMS,
        # A moderate class weight helps the trees find minority structure; Platt
        # calibration afterwards maps scores back to honest probabilities.
        scale_pos_weight=min(25.0, max(1.0, imbalance_ratio ** 0.5)),
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    pipeline = Pipeline([("features", TransactionFeatureBuilder()), ("classifier", booster)])
    return CalibratedClassifierCV(pipeline, method="sigmoid", cv=3)


CANDIDATES: list[Candidate] = [
    Candidate(
        version="v1",
        name="Logistic Regression baseline",
        family="Linear model",
        summary="Class-weighted logistic regression over the raw inputs. The original model.",
        highlights=["Raw 30 inputs", "Class-weighted", "Fully linear"],
        params={"class_weight": "balanced", "max_iter": 2000},
        build=_v1,
    ),
    Candidate(
        version="v2",
        name="Random Forest + behavioral features",
        family="Bagged trees",
        summary="Random forest over engineered time, amount and anomaly features.",
        highlights=["37 engineered features", "Non-linear interactions", "Balanced bootstrap"],
        params=V2_PARAMS,
        build=_v2,
    ),
    Candidate(
        version="v3",
        name="Calibrated LightGBM",
        family="Gradient-boosted trees",
        summary=(
            "Gradient boosting over engineered features, 3-fold cross-fitted with Platt "
            "calibration so scores are usable probabilities."
        ),
        highlights=["Gradient boosting", "3-fold calibration", "Reason codes", "Engineered features"],
        params={**V3_PARAMS, "calibration": "sigmoid", "calibration_folds": 3},
        build=_v3,
    ),
]
