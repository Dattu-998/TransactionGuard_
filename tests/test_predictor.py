import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.predictor import Predictor
from transactionguard.features.engineering import FeatureEngineer


@pytest.fixture
def mock_artifact_dir(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    model = LogisticRegression()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    model.fit(X, y)
    joblib.dump(model, artifact_dir / "model.joblib")
    fe = FeatureEngineer()
    df_train = pd.DataFrame({'F1': [1.0, 3.0], 'F2': [2.0, 4.0], 'Class': [0, 1]})
    fe.fit(df_train)
    joblib.dump(fe, artifact_dir / "feature_engineer.joblib")
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump({"model_name": "Logistic Regression", "version": "v1.0", "threshold": 0.5, "feature_names": ["F1", "F2"]}, f)
    with open(artifact_dir / "schema.json", "w") as f:
        json.dump({"F1": "float64", "F2": "float64"}, f)
    return artifact_dir

def test_predictor_load_success(mock_artifact_dir):
    predictor = Predictor(mock_artifact_dir)
    assert predictor.load() is True
    assert predictor.is_loaded is True
    assert predictor.get_info()["model_name"] == "Logistic Regression"

def test_predictor_load_failure(tmp_path):
    predictor = Predictor(tmp_path)
    assert predictor.load() is False
    assert predictor.is_loaded is False

def test_predictor_validate_features_valid(mock_artifact_dir):
    predictor = Predictor(mock_artifact_dir)
    predictor.load()
    is_valid, msg = predictor.validate_features(pd.DataFrame([{"F1": 1.5, "F2": 2.5}]))
    assert is_valid is True
    assert msg is None

def test_predictor_validate_features_missing(mock_artifact_dir):
    predictor = Predictor(mock_artifact_dir)
    predictor.load()
    is_valid, msg = predictor.validate_features(pd.DataFrame([{"F1": 1.5}]))
    assert is_valid is False
    assert "Missing required features" in msg

def test_predictor_validate_features_invalid_type(mock_artifact_dir):
    predictor = Predictor(mock_artifact_dir)
    predictor.load()
    is_valid, msg = predictor.validate_features(pd.DataFrame([{"F1": 1.5, "F2": "invalid_string"}]))
    assert is_valid is False
    assert "Non-numeric values" in msg

def test_predictor_predict(mock_artifact_dir):
    predictor = Predictor(mock_artifact_dir)
    predictor.load()
    predictions = predictor.predict(pd.DataFrame([{"F1": 1.5, "F2": 2.5}]))
    assert len(predictions) == 1
    assert "probability" in predictions[0]
    assert "is_fraud" in predictions[0]
    assert "threshold" in predictions[0]
    assert predictions[0]["threshold"] == 0.5
