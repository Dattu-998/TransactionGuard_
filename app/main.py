import logging
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

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
from transactionguard.monitoring.drift import calculate_drift

logger = logging.getLogger(__name__)

predictor_instance = Predictor(app_config.model_artifact_dir)

@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor_instance.load()
    app.state.predictor = predictor_instance
    yield
    app.state.predictor = None

app = FastAPI(title=app_config.app_name, lifespan=lifespan)

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
