import pandas as pd
from sklearn.model_selection import train_test_split

from transactionguard.config import config
from transactionguard.utils.logging import get_logger

logger = get_logger(__name__)

def split_data(df: pd.DataFrame, target_col: str = "Class"):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    relative_val_size = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=config.RANDOM_STATE, stratify=y_temp
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, val_df, test_df
