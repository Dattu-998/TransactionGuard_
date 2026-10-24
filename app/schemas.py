from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class ReadyResponse(BaseModel):
    ready: bool
    reason: str | None = None

class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    threshold: float
    features: dict[str, str]

class PredictionResponse(BaseModel):
    probability: float
    is_fraud: bool
    threshold: float
    model_version: str
    explanation: str

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    summary: dict[str, Any]

class ValidationResponse(BaseModel):
    valid: bool
    message: str | None = None
