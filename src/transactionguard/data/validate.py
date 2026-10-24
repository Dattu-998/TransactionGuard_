import numpy as np
import pandas as pd

from transactionguard.utils.logging import get_logger

logger = get_logger(__name__)

def validate_dataset(df: pd.DataFrame, target_col: str = "Class") -> bool:
    if df.empty:
        raise ValueError("Dataset is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    unique_classes = set(df[target_col].unique())
    if not unique_classes.issubset({0, 1}):
        raise ValueError("Target column must only contain 0 and 1.")
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[num_cols]).values.sum()
    if inf_count > 0:
        raise ValueError("Dataset contains infinite values.")
    return True

def validate_features(df: pd.DataFrame, expected_features: list[str]) -> bool:
    missing_features = [f for f in expected_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected features: {missing_features}")
    return True
