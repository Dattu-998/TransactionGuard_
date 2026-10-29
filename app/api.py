"""Versioned scoring, registry and monitoring endpoints mounted under ``/api``."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from app.config import app_config
from app.schemas import (
    BatchRequest,
    BatchResult,
    Comparison,
    Health,
    Prediction,
    TransactionIn,
)
from transactionguard.features.engineering import FEATURE_COLUMNS
from transactionguard.models.evaluate import evaluate_model, score_histogram
from transactionguard.models.registry import ModelRegistry
from transactionguard.monitoring.drift import drift_report
from transactionguard.scoring import ScoringError

router = APIRouter(prefix="/api")
LABEL_COLUMNS = ("Class", "is_fraud", "label")


def _registry(request: Request) -> ModelRegistry:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry is not loaded.")
    return registry


def _version(registry: ModelRegistry, version: str | None) -> str:
    try:
        return registry.get(version).version
    except ScoringError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _score(request: Request, frame: pd.DataFrame, version: str | None, explain_rows: str,
           record: bool = True):
    registry = _registry(request)
    resolved = _version(registry, version)
    try:
        results, reasons, elapsed = registry.score(frame, resolved, explain_rows=explain_rows)
    except ScoringError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record:
        request.app.state.telemetry.record_decisions(resolved, results["decision"])
    return resolved, results, reasons, elapsed


def _jsonable(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notnull(frame), None).to_dict(orient="records")


def _batch(request: Request, frame: pd.DataFrame, version: str | None) -> dict:
    if frame.columns.duplicated().any():
        raise HTTPException(status_code=400, detail="Duplicate column names are not supported.")
    explain_rows = "flagged" if len(frame) <= app_config.max_explained_rows * 4 else "none"
    resolved, results, reasons, elapsed = _score(request, frame, version, explain_rows)
    registry = _registry(request)

    decisions = results["decision"]
    flagged = decisions != "approve"
    amount = pd.to_numeric(frame["Amount"], errors="coerce").fillna(0)
    combined = pd.concat([frame.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
    rows = _jsonable(combined)
    for row, row_reasons in zip(rows, reasons, strict=True):
        row["reasons"] = row_reasons[:4]

    label_column = next((c for c in LABEL_COLUMNS if c in frame.columns), None)
    evaluation = None
    if label_column is not None:
        labels = pd.to_numeric(frame[label_column], errors="coerce")
        if labels.notna().all() and set(labels.unique()) <= {0, 1}:
            y = labels.astype(int).to_numpy()
            policy = registry.get(resolved).policy
            if len(np.unique(y)) == 2:
                evaluation = evaluate_model(y, results["fraud_probability"].to_numpy(),
                                            threshold=policy.review_threshold)
            else:
                pred = flagged.to_numpy()
                evaluation = {"tp": int((pred & (y == 1)).sum()), "fp": int((pred & (y == 0)).sum()),
                              "fn": int((~pred & (y == 1)).sum()), "tn": int((~pred & (y == 0)).sum())}
            evaluation["label_column"] = label_column
            evaluation["fraud_rows"] = int(y.sum())

    return {
        "model_version": resolved,
        "summary": {
            "processed": len(frame),
            "approve": int((decisions == "approve").sum()),
            "review": int((decisions == "review").sum()),
            "block": int((decisions == "block").sum()),
            "flagged": int(flagged.sum()),
            "flag_rate": float(flagged.mean()),
            "highest_risk": float(results["risk_score"].max()),
            "mean_risk": float(results["risk_score"].mean()),
            "amount_total": float(amount.sum()),
            "amount_flagged": float(amount[flagged.to_numpy()].sum()),
            "latency_ms": float(elapsed),
        },
        "score_histogram": score_histogram(np.zeros(len(results)), results["risk_score"]),
        "evaluation": evaluation,
        "rows": rows,
    }


# ---------------------------------------------------------------------- system
@router.get("/health", response_model=Health, tags=["system"])
async def health(request: Request):
    registry = getattr(request.app.state, "registry", None)
    return Health(
        status="ok",
        model_loaded=registry is not None,
        champion=registry.champion if registry else None,
        versions=registry.versions if registry else [],
        uptime_seconds=request.app.state.telemetry.uptime,
        api_version=app_config.api_version,
        load_errors=registry.load_errors if registry else {},
    )


@router.get("/metrics", tags=["system"])
async def metrics(request: Request):
    return request.app.state.telemetry.snapshot()


# -------------------------------------------------------------------- registry
@router.get("/models", tags=["registry"])
async def list_models(request: Request):
    return _registry(request).index


@router.get("/models/{version}", tags=["registry"])
async def model_card(version: str, request: Request):
    registry = _registry(request)
    return registry.get(_version(registry, version)).card


@router.get("/model-info", tags=["registry"])
async def model_info(request: Request):
    registry = _registry(request)
    card = registry.get().card
    return {"name": card["name"], "version": card["version"], "threshold": card["thresholds"]["review"],
            "thresholds": card["thresholds"], "features": FEATURE_COLUMNS, "metrics": card["test"],
            "evaluation_data": card["dataset"].get("source", "unknown")}


@router.get("/schema", tags=["registry"])
async def schema(request: Request):
    registry = _registry(request)
    return {"features": FEATURE_COLUMNS, "statistics": registry.feature_stats,
            "optional": ["transaction_id", *LABEL_COLUMNS]}


@router.get("/samples", tags=["registry"])
async def samples(
    request: Request,
    count: int = Query(20, ge=1, le=500),
    fraud_share: float = Query(0.08, ge=0, le=1),
    seed: int | None = None,
):
    """Held-out test transactions (never seen in training) for demos and the live feed."""
    registry = _registry(request)
    if registry.samples is None:
        raise HTTPException(status_code=404, detail="No demo transactions are bundled.")
    rng = np.random.default_rng(seed)
    pool = registry.samples
    fraud, legit = pool[pool["Class"] == 1], pool[pool["Class"] == 0]
    n_fraud = min(len(fraud), int(rng.binomial(count, fraud_share)))
    picks = pd.concat([
        fraud.sample(n=n_fraud, random_state=int(rng.integers(1 << 31))),
        legit.sample(n=min(len(legit), count - n_fraud), random_state=int(rng.integers(1 << 31))),
    ]).sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    return {"transactions": _jsonable(picks)}


# --------------------------------------------------------------------- scoring
@router.post("/predict", response_model=Prediction, tags=["scoring"])
async def predict(
    transaction: TransactionIn,
    request: Request,
    model: str | None = Query(None, description="Model version; defaults to the champion"),
    explain: bool = Query(True, description="Include reason codes"),
):
    payload = transaction.model_dump()
    frame = pd.DataFrame([payload])
    resolved, results, reasons, elapsed = _score(request, frame, model, "all" if explain else "none")
    row = results.iloc[0]
    return Prediction(
        transaction_id=payload.get("transaction_id"),
        model_version=resolved,
        fraud_probability=float(row["fraud_probability"]),
        risk_score=float(row["risk_score"]),
        risk_level=row["risk_level"],
        decision=row["decision"],
        thresholds={k: v for k, v in _registry(request).get(resolved).card["thresholds"].items()},
        reasons=reasons[0],
        latency_ms=elapsed,
    )


@router.post("/predict/compare", response_model=Comparison, tags=["scoring"])
async def compare(transaction: TransactionIn, request: Request):
    """Score one transaction with every registered version (champion/challenger view)."""
    registry = _registry(request)
    payload = transaction.model_dump()
    frame = pd.DataFrame([payload])
    results = []
    for version in registry.versions:
        _, scored, _, _ = _score(request, frame, version, "none")
        row = scored.iloc[0]
        entry = registry.get(version).card
        results.append({"version": version, "name": entry["name"], "stage": entry.get("stage", ""),
                        "fraud_probability": float(row["fraud_probability"]),
                        "risk_score": float(row["risk_score"]), "risk_level": row["risk_level"],
                        "decision": row["decision"]})
    return Comparison(transaction_id=payload.get("transaction_id"), champion=registry.champion,
                      results=results, agreement=len({r["decision"] for r in results}) == 1)


@router.post("/predict/batch", response_model=BatchResult, tags=["scoring"])
async def predict_batch(body: BatchRequest | list[dict[str, Any]], request: Request,
                        model: str | None = None):
    transactions = body.transactions if isinstance(body, BatchRequest) else body
    if not transactions:
        raise HTTPException(status_code=400, detail="The transaction batch is empty.")
    if len(transactions) > app_config.max_batch_size:
        raise HTTPException(status_code=413,
                            detail=f"Batch size exceeds the limit of {app_config.max_batch_size}.")
    return _batch(request, pd.DataFrame(transactions), model)


@router.post("/predict/upload", response_model=BatchResult, tags=["scoring"])
async def predict_upload(request: Request, file: UploadFile = File(...), model: str | None = None):
    name = (file.filename or "").lower()
    if not name.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Upload a CSV, XLSX, or XLS file.")
    content = await file.read()
    if len(content) > app_config.max_upload_bytes:
        raise HTTPException(status_code=413, detail="The upload exceeds the 25 MB safety limit.")
    try:
        stream = BytesIO(content)
        frame = pd.read_csv(stream) if name.endswith(".csv") else pd.read_excel(stream)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file could not be read.") from exc
    if len(frame) > app_config.max_upload_rows:
        raise HTTPException(status_code=413,
                            detail=f"Uploads are limited to {app_config.max_upload_rows:,} rows.")
    return _batch(request, frame, model)


# ------------------------------------------------------------------ monitoring
def _drift(request: Request, frame: pd.DataFrame) -> dict:
    registry = _registry(request)
    if not registry.drift_reference:
        raise HTTPException(status_code=503, detail="No drift reference is bundled with the registry.")
    _, results, _, _ = _score(request, frame, None, "none", record=False)
    current = frame.copy()
    current["risk_score"] = results["risk_score"].to_numpy()
    report = drift_report(registry.drift_reference, current)
    report["flag_rate"] = float((results["decision"] != "approve").mean())
    return report


@router.post("/monitor/drift", tags=["monitoring"])
async def monitor_drift(body: BatchRequest | list[dict[str, Any]], request: Request):
    transactions = body.transactions if isinstance(body, BatchRequest) else body
    if len(transactions) < 50:
        raise HTTPException(status_code=400, detail="Drift checks need at least 50 transactions.")
    return _drift(request, pd.DataFrame(transactions[: app_config.max_batch_size]))


SCENARIOS = {
    "baseline": "Held-out traffic from the same period as training.",
    "holiday_surge": "Seasonal spend: amounts up ~80%, activity shifted to evenings.",
    "fraud_ring": "A coordinated attack: account-takeover traffic at ~15x the usual fraud rate.",
    "schema_shift": "An upstream pipeline change rescales two behavioral components.",
}


@router.get("/monitor/drift/scenarios", tags=["monitoring"])
async def drift_scenarios():
    return SCENARIOS


@router.get("/monitor/drift/simulate", tags=["monitoring"])
async def simulate_drift(request: Request, scenario: str = "baseline"):
    """Run the drift monitor on a simulated production window (for demos)."""
    if scenario not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown scenario. Choose from {list(SCENARIOS)}.")
    registry = _registry(request)
    if registry.samples is None:
        raise HTTPException(status_code=404, detail="No demo transactions are bundled.")
    rng = np.random.default_rng(7)
    pool = registry.samples[FEATURE_COLUMNS + ["Class"]]
    legit, fraud_pool = pool[pool["Class"] == 0], pool[pool["Class"] == 1]
    # The demo pool is fraud-enriched; restore production prevalence for the window.
    rate = float(registry.index.get("dataset", {}).get("fraud_rate", 0.002))
    n_fraud = min(len(fraud_pool), max(1, round(len(legit) * rate / (1 - rate))))
    window = pd.concat([legit, fraud_pool.sample(n=n_fraud, random_state=7)], ignore_index=True)
    if scenario == "holiday_surge":
        window["Amount"] *= rng.lognormal(0.6, 0.25, len(window))
        window["Time"] = (window["Time"] + 4 * 3600) % (2 * 86_400)
    elif scenario == "fraud_ring":
        extra = fraud_pool.sample(n=len(fraud_pool), random_state=7).copy()
        for column, shift in {"V14": -1.2, "V17": -0.9, "V10": -0.7}.items():
            extra[column] += shift + rng.normal(0, 0.3, len(extra))
        window = pd.concat([window, extra], ignore_index=True)
    elif scenario == "schema_shift":
        window["V3"] = window["V3"] * 1.9 + 0.6
        window["V12"] = window["V12"] * 0.4 - 0.8
    report = _drift(request, window.drop(columns=["Class"]))
    report["scenario"] = scenario
    report["description"] = SCENARIOS[scenario]
    return report
