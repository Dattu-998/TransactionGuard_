import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.api import router
from app.config import app_config
from app.telemetry import Telemetry
from transactionguard.models.registry import ModelRegistry
from transactionguard.scoring import ScoringError

logger = logging.getLogger("transactionguard.api")

DEFAULT_ORIGINS = [
    "https://transactionguard-frontend-h7njdhmeq-dattu-998.vercel.app",
    "https://transactionguard-frontend-neon.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def load_registry(artifact_dir: Path) -> ModelRegistry | None:
    try:
        return ModelRegistry(artifact_dir).load()
    except ScoringError as exc:
        logger.error("Model registry unavailable: %s", exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.telemetry = Telemetry()
    app.state.registry = load_registry(app_config.model_artifact_dir)
    yield
    app.state.registry = None


app = FastAPI(
    title=app_config.app_name,
    version=app_config.api_version,
    description=(
        "Versioned fraud-scoring service: champion/challenger model registry, calibrated "
        "risk scores with approve/review/block decisions, reason codes, batch scoring, "
        "and PSI drift monitoring."
    ),
    lifespan=lifespan,
)

allowed_origins = DEFAULT_ORIGINS + [
    origin.strip() for origin in app_config.cors_allowed_origins.split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(allowed_origins)),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    telemetry = getattr(request.app.state, "telemetry", None)
    if telemetry is not None and request.method != "OPTIONS":
        telemetry.record_request(request.url.path, response.status_code, elapsed_ms)
    return response


app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    return {"service": app_config.app_name, "version": app_config.api_version, "docs": "/docs"}


@app.get("/health", tags=["system"])
async def health_check(request: Request):
    return {"status": "ok", "model_loaded": getattr(request.app.state, "registry", None) is not None}


@app.get("/ready", tags=["system"])
async def readiness_check(request: Request):
    if getattr(request.app.state, "registry", None) is None:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            content={"ready": False, "reason": "Model registry not loaded."})
    return {"ready": True}
