from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from transactionguard.features.engineering import FEATURE_COLUMNS

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

_EXAMPLE = {
    "transaction_id": "TX-100042",
    "Time": 7_310.0,
    **{f"V{i}": 0.0 for i in range(1, 29)},
    "Amount": 129.99,
}
_EXAMPLE.update({"V4": 2.1, "V12": -1.9, "V14": -3.6, "V17": -2.8})

TransactionIn = create_model(
    "TransactionIn",
    __config__=ConfigDict(extra="allow", json_schema_extra={"example": _EXAMPLE}),
    transaction_id=(str | None, None),
    **{name: (FiniteFloat, ...) for name in FEATURE_COLUMNS},
)

Decision = Literal["approve", "review", "block"]
RiskLevel = Literal["low", "elevated", "high", "critical"]


class Reason(BaseModel):
    feature: str
    label: str
    value: float
    baseline: float
    contribution: float = Field(description="Change in fraud log-odds versus the typical value")
    direction: Literal["increases_risk", "decreases_risk"]


class Prediction(BaseModel):
    transaction_id: str | None = None
    model_version: str
    fraud_probability: float
    risk_score: float = Field(description="0-100; 50 = review threshold, 80 = block threshold")
    risk_level: RiskLevel
    decision: Decision
    thresholds: dict[str, float]
    reasons: list[Reason] = []
    latency_ms: float


class ComparisonResult(BaseModel):
    version: str
    name: str
    stage: str
    fraud_probability: float
    risk_score: float
    risk_level: RiskLevel
    decision: Decision


class Comparison(BaseModel):
    transaction_id: str | None = None
    champion: str
    results: list[ComparisonResult]
    agreement: bool


class BatchRequest(BaseModel):
    transactions: list[dict[str, Any]]


class BatchSummary(BaseModel):
    processed: int
    approve: int
    review: int
    block: int
    flagged: int
    flag_rate: float
    highest_risk: float
    mean_risk: float
    amount_total: float
    amount_flagged: float
    latency_ms: float


class BatchResult(BaseModel):
    model_version: str
    summary: BatchSummary
    score_histogram: list[dict[str, Any]]
    evaluation: dict[str, Any] | None = None
    rows: list[dict[str, Any]]


class Health(BaseModel):
    status: str
    model_loaded: bool
    champion: str | None = None
    versions: list[str] = []
    uptime_seconds: float
    api_version: str
    load_errors: dict[str, str] = {}
