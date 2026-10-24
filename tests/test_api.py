import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app.config import app_config
from app.main import app, predictor_instance
from transactionguard.features.engineering import FeatureEngineer


@pytest.fixture(autouse=True)
def mock_artifacts(tmp_path):
    app_config.model_artifact_dir = tmp_path
    predictor_instance.artifact_dir = tmp_path
    model = LogisticRegression()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    model.fit(X, y)
    joblib.dump(model, tmp_path / "model.joblib")
    fe = FeatureEngineer()
    df_train = pd.DataFrame({'F1': [1.0, 3.0], 'F2': [2.0, 4.0], 'Class': [0, 1]})
    fe.fit(df_train)
    joblib.dump(fe, tmp_path / "feature_engineer.joblib")
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump({"model_name": "TestModel", "version": "v1.0", "threshold": 0.5, "feature_names": ["F1", "F2"]}, f)
    with open(tmp_path / "schema.json", "w") as f:
        json.dump({"F1": "float64", "F2": "float64"}, f)
    return tmp_path

def test_health_check_with_model():
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] is True

def test_readiness_check_with_model():
    with TestClient(app) as test_client:
        response = test_client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

def test_model_info():
    with TestClient(app) as test_client:
        response = test_client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "TestModel"
        assert data["version"] == "v1.0"
        assert "F1" in data["features"]

def test_predict_single_valid():
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"F1": 1.5, "F2": 2.5})
        assert response.status_code == 200
        data = response.json()
        assert "probability" in data
        assert "is_fraud" in data
        assert data["threshold"] == 0.5

def test_predict_single_invalid_missing():
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"F1": 1.5})
        assert response.status_code == 400
        assert "Missing required features" in response.json()["detail"]

def test_predict_single_invalid_extra():
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"F1": 1.5, "F2": 2.5, "F3": 3.5})
        assert response.status_code == 400
        assert "Unexpected features found" in response.json()["detail"]

def test_predict_single_invalid_nan():
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"F1": 1.5, "F2": "NaN"})
        assert response.status_code == 400
        assert "Non-numeric values" in response.json()["detail"] or "Missing values" in response.json()["detail"]

def test_predict_single_invalid_inf():
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"F1": 1.5, "F2": "Infinity"})
        assert response.status_code == 400
        assert "Non-numeric values" in response.json()["detail"] or "Infinite values" in response.json()["detail"]

def test_predict_batch_valid():
    with TestClient(app) as test_client:
        response = test_client.post("/predict/batch", json=[{"F1": 1.5, "F2": 2.5}, {"F1": -1.5, "F2": -2.5}])
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2
        assert data["summary"]["batch_size"] == 2

def test_validate_endpoint():
    with TestClient(app) as test_client:
        response = test_client.post("/validate", json={"F1": 1.5, "F2": 2.5})
        assert response.status_code == 200
        assert response.json()["valid"] is True

def test_drift_valid(mock_artifacts, monkeypatch):
    from transactionguard.config import config

    reference_path = config.REFERENCE_DATA_DIR / "reference.csv"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    df_train = pd.DataFrame({'F1': [1.0, 3.0], 'F2': [2.0, 4.0], 'Class': [0, 1]})
    df_train.to_csv(reference_path, index=False)

    with TestClient(app) as test_client:
        payload = [
            {"F1": 1.5, "F2": 2.5},
            {"F1": -1.5, "F2": -2.5},
            {"F1": 1.0, "F2": 2.0}
        ]
        response = test_client.post("/monitor/drift", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "features_checked" in data
        assert "feature_details" in data
        assert "F1" in data["feature_details"]
        assert "statistic" in data["feature_details"]["F1"]

def test_drift_invalid_missing_ref(mock_artifacts):
    from transactionguard.config import config
    reference_path = config.REFERENCE_DATA_DIR / "reference.csv"
    if reference_path.exists():
        reference_path.unlink()

    with TestClient(app) as test_client:
        payload = [
            {"F1": 1.5, "F2": 2.5}
        ]
        response = test_client.post("/monitor/drift", json=payload)
        assert response.status_code == 503
        assert "Reference data not found" in response.json()["detail"]

def test_drift_invalid_data(mock_artifacts):
    with TestClient(app) as test_client:
        payload = [
            {"F1": 1.5}
        ]
        response = test_client.post("/monitor/drift", json=payload)
        assert response.status_code == 400
        assert "Missing required features" in response.json()["detail"]
