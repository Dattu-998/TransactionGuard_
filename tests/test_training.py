import numpy as np
import pandas as pd
import pytest

from transactionguard.models.train import (
    train_lightgbm,
    train_logistic_regression,
    train_random_forest,
)


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X = pd.DataFrame({'F1': np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 10)]), 'F2': np.concatenate([np.random.normal(0, 1, 100), np.random.normal(-5, 1, 10)])})
    y = pd.Series(np.concatenate([np.zeros(100), np.ones(10)]))
    return X, y

def test_train_logistic_regression(dummy_data):
    X, y = dummy_data
    model = train_logistic_regression(X, y)
    assert hasattr(model, "predict_proba")
    probs = model.predict_proba(X)
    assert probs.shape == (len(y), 2)
    assert (probs >= 0).all() and (probs <= 1).all()

def test_train_random_forest(dummy_data):
    X, y = dummy_data
    model = train_random_forest(X, y)
    assert hasattr(model, "predict_proba")
    probs = model.predict_proba(X)
    assert probs.shape == (len(y), 2)

def test_train_lightgbm(dummy_data):
    X, y = dummy_data
    model = train_lightgbm(X, y)
    assert hasattr(model, "predict_proba")
    probs = model.predict_proba(X)
    assert probs.shape == (len(y), 2)
