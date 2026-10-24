import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from transactionguard.utils.logging import get_logger

logger = get_logger(__name__)

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series):
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    model = lgb.LGBMClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model
