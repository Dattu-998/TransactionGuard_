import logging
import os
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.config import app_config
from app.dependencies import get_predictor
from app.predictor import Predictor
from app.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadyResponse,
    ValidationResponse,
)
from transactionguard.config import config
from transactionguard.inference import InferenceError, InferenceService
from transactionguard.monitoring.drift import calculate_drift

logger = logging.getLogger(__name__)

predictor_instance = Predictor(app_config.model_artifact_dir)

@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor_instance.load()
    app.state.predictor = predictor_instance
    try:
        app.state.inference = InferenceService(app_config.model_artifact_dir).load()
    except InferenceError:
        app.state.inference = None
    yield
    app.state.predictor = None
    app.state.inference = None

app = FastAPI(title=app_config.app_name, lifespan=lifespan)
allowed_origins = [
    "https://transactionguard-frontend-h7njdhmeq-dattu-998.vercel.app",
    "https://transactionguard-frontend-neon.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
allowed_origins.extend(
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(allowed_origins)),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _service() -> InferenceService:
    service = getattr(app.state, "inference", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return service


def _result_payload(source: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    result = pd.concat([source.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    return {
        "rows": result.where(pd.notnull(result), None).to_dict(orient="records"),
        "summary": {
            "processed": len(result),
            "flagged": int((predictions["Predicted Class"] == "Potential Fraud").sum()),
            "highest_probability": float(predictions["Fraud Probability"].max()) if len(result) else 0.0,
        },
    }


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "model_loaded": getattr(app.state, "inference", None) is not None}


@app.get("/api/model-info")
async def api_model_info():
    service = _service()
    metrics_path = config.REPORT_DIR / "test_metrics.json"
    metrics = pd.read_json(metrics_path, typ="series").to_dict() if metrics_path.exists() else {}
    return {"name": service.details.name, "version": service.details.version,
            "threshold": service.details.threshold, "features": service.details.feature_names,
            "metrics": metrics, "evaluation_data": "synthetic"}


@app.post("/api/predict")
async def api_predict(transaction: dict[str, Any]):
    service = _service()
    try:
        prediction = service.predict(pd.DataFrame([transaction])).iloc[0].to_dict()
    except InferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return prediction


@app.post("/api/predict/batch")
async def api_predict_batch(transactions: list[dict[str, Any]]):
    if not transactions:
        raise HTTPException(status_code=400, detail="The transaction batch is empty.")
    if len(transactions) > app_config.max_batch_size:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds limit ({app_config.max_batch_size}).")
    service = _service()
    frame = pd.DataFrame(transactions)
    try:
        return _result_payload(frame, service.predict(frame))
    except InferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/predict/upload")
async def api_predict_upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Upload a CSV, XLSX, or XLS file.")
    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The upload exceeds the 25 MB safety limit.")
    try:
        stream = BytesIO(content)
        frame = pd.read_csv(stream) if file.filename.lower().endswith(".csv") else pd.read_excel(stream)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file could not be read.") from exc
    if frame.columns.duplicated().any():
        raise HTTPException(status_code=400, detail="Duplicate column names are not supported.")
    service = _service()
    try:
        return _result_payload(frame, service.predict(frame))
    except InferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", model_loaded=app.state.predictor.is_loaded if hasattr(app.state, 'predictor') else False)

@app.get("/ready", response_model=ReadyResponse)
async def readiness_check():
    if not hasattr(app.state, 'predictor') or not app.state.predictor.is_loaded:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"ready": False, "reason": "Model artifacts not loaded."})
    return ReadyResponse(ready=True)

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info(predictor: Predictor = Depends(get_predictor)):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return predictor.get_info()

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(transaction: dict[str, Any], predictor: Predictor = Depends(get_predictor)):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    df = pd.DataFrame([transaction])
    is_valid, msg = predictor.validate_features(df)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)
    try:
        return predictor.predict(df)[0]
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Internal prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(transactions: list[dict[str, Any]], predictor: Predictor = Depends(get_predictor)):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not transactions:
        raise HTTPException(status_code=400, detail="Empty batch.")
    if len(transactions) > app_config.max_batch_size:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds limit ({app_config.max_batch_size}).")
    df = pd.DataFrame(transactions)
    is_valid, msg = predictor.validate_features(df)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)
    try:
        preds = predictor.predict(df)
        fraud_count = sum(1 for p in preds if p["is_fraud"])
        summary = {"batch_size": len(preds), "fraud_detected": fraud_count, "fraud_prevalence": fraud_count / len(preds)}
        return BatchPredictionResponse(predictions=preds, summary=summary)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Internal batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

@app.post("/validate", response_model=ValidationResponse)
async def validate_transaction(transaction: dict[str, Any], predictor: Predictor = Depends(get_predictor)):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model schema not available.")
    df = pd.DataFrame([transaction])
    is_valid, msg = predictor.validate_features(df)
    return ValidationResponse(valid=is_valid, message=msg)

@app.post("/monitor/drift")
async def check_drift(current_data: list[dict[str, Any]], predictor: Predictor = Depends(get_predictor)):
    if not current_data:
        raise HTTPException(status_code=400, detail="Empty data.")

    curr_df = pd.DataFrame(current_data)

    if predictor.is_loaded:
        is_valid, msg = predictor.validate_features(curr_df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=msg)

    ref_path = config.REFERENCE_DATA_DIR / "reference.csv"
    if not ref_path.exists():
        raise HTTPException(status_code=503, detail="Reference data not found. Cannot calculate drift.")

    try:
        ref_df = pd.read_csv(ref_path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        logger.error(f"Failed to read reference data: {e}")
        raise HTTPException(status_code=500, detail="Error loading reference data.")

    num_features = [col for col in curr_df.columns if pd.api.types.is_numeric_dtype(curr_df[col])]

    try:
        summary = calculate_drift(ref_df, curr_df, num_features)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Failed to calculate drift: {e}")
        raise HTTPException(status_code=500, detail="Internal drift calculation error.")

    return summary
