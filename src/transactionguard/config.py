import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_PATH: Path = DATA_DIR / "raw" / "creditcard.csv"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    REFERENCE_DATA_DIR: Path = DATA_DIR / "reference"
    ARTIFACT_DIR: Path = Path(os.getenv("MODEL_ARTIFACT_DIR", BASE_DIR / "artifacts"))
    REPORT_DIR: Path = BASE_DIR / "reports"

    TARGET_COL: str = "Class"
    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.15
    RANDOM_STATE: int = 42

    FN_COST: float = 1000.0
    FP_COST: float = 10.0

config = Config()
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
(config.DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
config.REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
