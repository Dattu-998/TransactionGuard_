import os
from pathlib import Path

from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    app_name: str = "TransactionGuard API"
    model_artifact_dir: Path = Path(os.getenv("MODEL_ARTIFACT_DIR", "./artifacts"))
    max_batch_size: int = 1000
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

app_config = AppConfig()
