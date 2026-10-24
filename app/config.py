from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parent.parent


class AppConfig(BaseSettings):
    app_name: str = "TransactionGuard API"
    model_artifact_dir: Path = ROOT / "artifacts"
    max_batch_size: int = 1000
    log_level: str = "INFO"

    @field_validator("model_artifact_dir", mode="before")
    @classmethod
    def resolve_model_artifact_dir(cls, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        return path.resolve()

app_config = AppConfig()
