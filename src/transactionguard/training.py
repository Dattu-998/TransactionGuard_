"""End-to-end training: fit every candidate version, evaluate, and publish the registry."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from transactionguard.config import config
from transactionguard.data.split import split_data
from transactionguard.data.validate import validate_dataset
from transactionguard.features.engineering import FEATURE_COLUMNS, feature_label, feature_statistics
from transactionguard.models.candidates import CANDIDATES, Candidate
from transactionguard.models.evaluate import (
    calibration_points,
    evaluate_model,
    pr_curve_points,
    roc_curve_points,
    score_histogram,
)
from transactionguard.models.registry import REGISTRY_FILE
from transactionguard.models.threshold import (
    optimize_threshold,
    precision_threshold,
    threshold_sweep,
)
from transactionguard.monitoring.drift import build_reference
from transactionguard.scoring import DecisionPolicy, positive_probabilities
from transactionguard.utils.logging import get_logger

logger = get_logger(__name__)

BLOCK_PRECISION_TARGET = 0.90
DEMO_LEGIT_ROWS = 2500


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=config.BASE_DIR, timeout=5, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _latency(estimator, sample: pd.DataFrame) -> dict:
    single = sample.iloc[[0]]
    positive_probabilities(estimator, single)  # warm-up
    timings = []
    for i in range(60):
        row = sample.iloc[[i % len(sample)]]
        started = time.perf_counter()
        positive_probabilities(estimator, row)
        timings.append((time.perf_counter() - started) * 1000)
    batch = sample.iloc[:1000]
    started = time.perf_counter()
    positive_probabilities(estimator, batch)
    batch_seconds = time.perf_counter() - started
    return {
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "batch_throughput_per_sec": float(len(batch) / max(batch_seconds, 1e-9)),
    }


def _importance(estimator, X: pd.DataFrame, y: pd.Series) -> list[dict]:
    result = permutation_importance(
        estimator, X, y, scoring="average_precision", n_repeats=3,
        random_state=config.RANDOM_STATE, n_jobs=1,
    )
    rows = [
        {"feature": name, "label": feature_label(name),
         "importance": float(mean), "std": float(std)}
        for name, mean, std in zip(X.columns, result.importances_mean, result.importances_std,
                                   strict=True)
    ]
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return rows


def _business(y_true: np.ndarray, flagged: np.ndarray, amounts: np.ndarray) -> dict:
    positives = int(y_true.sum())
    tp = int((flagged & (y_true == 1)).sum())
    fp = int((flagged & (y_true == 0)).sum())
    cost = (positives - tp) * config.FN_COST + fp * config.FP_COST
    no_model_cost = positives * config.FN_COST
    fraud_amount = float(amounts[y_true == 1].sum())
    caught_amount = float(amounts[flagged & (y_true == 1)].sum())
    return {
        "expected_cost": float(cost),
        "no_model_cost": float(no_model_cost),
        "cost_reduction": float(1 - cost / no_model_cost) if no_model_cost else 0.0,
        "fraud_amount": fraud_amount,
        "fraud_amount_caught": caught_amount,
        "amount_recall": caught_amount / fraud_amount if fraud_amount else 0.0,
        "alerts_per_1000": float(1000 * flagged.mean()),
    }


def train_candidate(candidate: Candidate, splits: dict, imbalance_ratio: float) -> tuple[object, dict]:
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    logger.info("Training %s - %s", candidate.version, candidate.name)

    estimator = candidate.build(imbalance_ratio)
    started = time.perf_counter()
    estimator.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started

    val_prob = positive_probabilities(estimator, X_val)
    test_prob = positive_probabilities(estimator, X_test)

    review, review_metrics = optimize_threshold(y_val.to_numpy(), val_prob)
    block = precision_threshold(y_val.to_numpy(), val_prob, BLOCK_PRECISION_TARGET, floor=review)
    policy = DecisionPolicy(review, block)

    val_eval = evaluate_model(y_val, val_prob, threshold=review)
    test_eval = evaluate_model(y_test, test_prob, threshold=review)
    test_block = evaluate_model(y_test, test_prob, threshold=block)
    decisions = policy.decisions(test_prob)
    y_test_np = y_test.to_numpy()

    rng = np.random.default_rng(config.RANDOM_STATE)
    legit_idx = np.flatnonzero(y_val.to_numpy() == 0)
    keep = np.r_[np.flatnonzero(y_val.to_numpy() == 1),
                 rng.choice(legit_idx, size=min(15_000, len(legit_idx)), replace=False)]
    importance = _importance(estimator, X_val.iloc[keep], y_val.iloc[keep])

    card = {
        "version": candidate.version,
        "name": candidate.name,
        "family": candidate.family,
        "summary": candidate.summary,
        "highlights": candidate.highlights,
        "params": candidate.params,
        "thresholds": {"review": review, "block": block,
                       "block_precision_target": BLOCK_PRECISION_TARGET},
        "training": {"seconds": train_seconds, "rows": len(X_train),
                     "fraud_rows": int(y_train.sum())},
        "validation": {**val_eval, "cost_optimal": review_metrics},
        "test": test_eval,
        "test_block": test_block,
        "decision_mix": {d: float((decisions == d).mean()) for d in ("approve", "review", "block")},
        "business": _business(y_test_np, test_prob >= review, X_test["Amount"].to_numpy()),
        "curves": {
            "pr": pr_curve_points(y_test, test_prob),
            "roc": roc_curve_points(y_test, test_prob),
            "calibration": calibration_points(y_test, test_prob),
            "threshold_sweep": threshold_sweep(y_val.to_numpy(), val_prob),
            "score_histogram": score_histogram(y_test_np, policy.risk_scores(test_prob)),
        },
        "importance": importance,
        "latency": _latency(estimator, X_test),
    }
    logger.info("%s validation PR-AUC %.4f | test PR-AUC %.4f | recall %.3f precision %.3f",
                candidate.version, val_eval["pr_auc"], test_eval["pr_auc"],
                test_eval["recall"], test_eval["precision"])
    return estimator, card


def _assign_stages(cards: dict[str, dict]) -> str:
    ranked = sorted(cards, key=lambda v: cards[v]["validation"]["pr_auc"], reverse=True)
    for position, version in enumerate(ranked):
        cards[version]["stage"] = ("champion", "challenger")[min(position, 1)] if position < 2 else "archived"
    return ranked[0]


def _demo_samples(test_df: pd.DataFrame) -> pd.DataFrame:
    fraud = test_df[test_df[config.TARGET_COL] == 1]
    legit = test_df[test_df[config.TARGET_COL] == 0].sample(
        n=min(DEMO_LEGIT_ROWS, int((test_df[config.TARGET_COL] == 0).sum())),
        random_state=config.RANDOM_STATE,
    )
    sample = pd.concat([legit, fraud]).sample(frac=1.0, random_state=config.RANDOM_STATE)
    sample.insert(0, "transaction_id", [f"TX-{100000 + i}" for i in range(len(sample))])
    return sample.reset_index(drop=True)


def train_registry(df: pd.DataFrame, artifact_dir: Path, data_info: dict,
                   versions: list[str] | None = None, processed_dir: Path | None = None) -> dict:
    validate_dataset(df)
    train_df, val_df, test_df = split_data(df)
    target = config.TARGET_COL
    splits = {name: (part[FEATURE_COLUMNS], part[target])
              for name, part in (("train", train_df), ("val", val_df), ("test", test_df))}
    y_train = splits["train"][1]
    imbalance_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    models_dir = artifact_dir / "models"
    cards: dict[str, dict] = {}
    for candidate in CANDIDATES:
        if versions and candidate.version not in versions:
            continue
        estimator, card = train_candidate(candidate, splits, imbalance_ratio)
        folder = models_dir / candidate.version
        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / "model.joblib"
        joblib.dump(estimator, model_path, compress=3)
        card["artifact_bytes"] = model_path.stat().st_size
        cards[candidate.version] = card

    champion = _assign_stages(cards)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    dataset = {
        **data_info,
        "rows": len(df),
        "fraud_rows": int(df[target].sum()),
        "fraud_rate": float(df[target].mean()),
        "splits": {name: {"rows": len(part), "fraud_rows": int(part[target].sum())}
                   for name, part in (("train", train_df), ("validation", val_df), ("test", test_df))},
    }
    environment = {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                   "git_commit": _git_commit()}
    for version, card in cards.items():
        card.update({"created_at": created_at, "dataset": dataset, "environment": environment,
                     "costs": {"false_negative": config.FN_COST, "false_positive": config.FP_COST}})
        (models_dir / version / "card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    index = {
        "champion": champion,
        "created_at": created_at,
        "dataset": dataset,
        "costs": {"false_negative": config.FN_COST, "false_positive": config.FP_COST},
        "versions": [
            {
                "version": v,
                "name": c["name"],
                "family": c["family"],
                "stage": c["stage"],
                "summary": c["summary"],
                "highlights": c["highlights"],
                "created_at": created_at,
                "thresholds": c["thresholds"],
                "metrics": {
                    "val_pr_auc": c["validation"]["pr_auc"],
                    "pr_auc": c["test"]["pr_auc"],
                    "roc_auc": c["test"]["roc_auc"],
                    "precision": c["test"]["precision"],
                    "recall": c["test"]["recall"],
                    "f1": c["test"]["f1"],
                    "brier": c["test"]["brier"],
                    "fp": c["test"]["fp"],
                    "fn": c["test"]["fn"],
                    "cost_reduction": c["business"]["cost_reduction"],
                    "amount_recall": c["business"]["amount_recall"],
                    "latency_p50_ms": c["latency"]["p50_ms"],
                },
                "artifact_bytes": c["artifact_bytes"],
            }
            for v, c in cards.items()
        ],
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / REGISTRY_FILE).write_text(json.dumps(index, indent=2), encoding="utf-8")
    (artifact_dir / "feature_stats.json").write_text(
        json.dumps(feature_statistics(train_df), indent=2), encoding="utf-8")

    champion_model = joblib.load(models_dir / champion / "model.joblib")
    thresholds = cards[champion]["thresholds"]
    champion_policy = DecisionPolicy(thresholds["review"], thresholds["block"])
    reference = {name: build_reference(train_df[name]) for name in FEATURE_COLUMNS}
    val_scores = champion_policy.risk_scores(positive_probabilities(champion_model, splits["val"][0]))
    reference["risk_score"] = build_reference(val_scores)
    (artifact_dir / "drift_reference.json").write_text(json.dumps(reference), encoding="utf-8")

    samples_dir = artifact_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    _demo_samples(test_df).round(6).to_csv(samples_dir / "demo_transactions.csv", index=False)
    if processed_dir is not None:
        test_df.to_csv(processed_dir / "test_data.csv", index=False)
    logger.info("Registry published to %s (champion: %s)", artifact_dir, champion)
    return index


def dataset_info(path: Path) -> dict:
    sidecar = path.with_suffix(".source.json")
    info = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {
        "source": "external", "description": "Kaggle-schema credit card transactions"}
    return {**info, "fingerprint": _fingerprint(path)}
