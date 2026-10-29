import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import app_config
from app.main import app
from transactionguard.features.engineering import FEATURE_COLUMNS

ALLOWED_CORS_ORIGINS = [
    "https://transactionguard-frontend-h7njdhmeq-dattu-998.vercel.app",
    "https://transactionguard-frontend-neon.vercel.app",
    "http://localhost:5173",
]


@pytest.fixture
def client(registry_dir, monkeypatch):
    monkeypatch.setattr(app_config, "model_artifact_dir", registry_dir)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def empty_client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_config, "model_artifact_dir", tmp_path)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def fraud_row(benchmark):
    return benchmark[benchmark["Class"] == 1].iloc[0][FEATURE_COLUMNS].to_dict()


@pytest.mark.parametrize("origin", ALLOWED_CORS_ORIGINS)
@pytest.mark.parametrize("path", ["/api/health", "/api/models"])
def test_cors_for_allowed_origins(client, origin, path):
    preflight = client.options(path, headers={"Origin": origin, "Access-Control-Request-Method": "GET"})
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == origin
    response = client.get(path, headers={"Origin": origin})
    assert response.headers["access-control-allow-origin"] == origin


def test_health_and_readiness(client):
    health = client.get("/api/health").json()
    assert health["model_loaded"] is True
    assert health["versions"] == ["v1", "v2", "v3"]
    assert client.get("/health").json()["model_loaded"] is True
    assert client.get("/ready").json() == {"ready": True}


def test_unloaded_registry_is_reported(empty_client):
    assert empty_client.get("/api/health").json()["model_loaded"] is False
    assert empty_client.get("/ready").status_code == 503
    assert empty_client.get("/api/models").status_code == 503


def test_registry_endpoints(client):
    index = client.get("/api/models").json()
    champion = index["champion"]
    card = client.get(f"/api/models/{champion}").json()
    assert card["stage"] == "champion"
    assert client.get("/api/models/v99").status_code == 404
    schema = client.get("/api/schema").json()
    assert schema["features"] == FEATURE_COLUMNS
    assert "median" in schema["statistics"]["Amount"]


def test_samples(client):
    rows = client.get("/api/samples?count=30&fraud_share=0.5&seed=3").json()["transactions"]
    assert len(rows) == 30
    assert {"transaction_id", "Class", *FEATURE_COLUMNS} <= set(rows[0])


def test_predict_returns_decision_and_reasons(client, fraud_row):
    response = client.post("/api/predict", json={"transaction_id": "T-1", **fraud_row})
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "T-1"
    assert data["decision"] in {"approve", "review", "block"}
    assert 0 <= data["risk_score"] <= 100
    assert data["reasons"] and data["reasons"][0]["feature"] in FEATURE_COLUMNS
    assert "X-Request-ID" in response.headers


def test_predict_specific_version(client, fraud_row):
    data = client.post("/api/predict?model=v1&explain=false", json=fraud_row).json()
    assert data["model_version"] == "v1" and data["reasons"] == []
    assert client.post("/api/predict?model=v42", json=fraud_row).status_code == 404


@pytest.mark.parametrize("change", [{"V1": "abc"}, {"Amount": None}])
def test_predict_rejects_invalid_values(client, fraud_row, change):
    assert client.post("/api/predict", json={**fraud_row, **change}).status_code == 422


def test_predict_rejects_missing_features(client):
    response = client.post("/api/predict", json={"Time": 1.0, "Amount": 3.0})
    assert response.status_code == 422


def test_compare_scores_every_version(client, fraud_row):
    data = client.post("/api/predict/compare", json=fraud_row).json()
    assert [r["version"] for r in data["results"]] == ["v1", "v2", "v3"]
    assert isinstance(data["agreement"], bool)


def test_batch_with_labels_includes_evaluation(client, benchmark):
    batch = benchmark.groupby("Class").head(40)
    data = client.post("/api/predict/batch", json={"transactions": batch.to_dict("records")}).json()
    summary = data["summary"]
    assert summary["processed"] == 80
    assert summary["approve"] + summary["review"] + summary["block"] == 80
    assert data["evaluation"]["fraud_rows"] == 40
    assert len(data["rows"]) == 80 and "decision" in data["rows"][0]


def test_batch_limits(client):
    assert client.post("/api/predict/batch", json=[]).status_code == 400


def test_upload_csv(client, benchmark):
    buffer = io.StringIO()
    benchmark.head(25).drop(columns="Class").assign(customer="c").to_csv(buffer, index=False)
    files = {"file": ("batch.csv", buffer.getvalue(), "text/csv")}
    data = client.post("/api/predict/upload?model=v2", files=files).json()
    assert data["model_version"] == "v2"
    assert data["evaluation"] is None
    assert data["rows"][0]["customer"] == "c"


def test_upload_rejects_bad_files(client):
    files = {"file": ("notes.txt", "hello", "text/plain")}
    assert client.post("/api/predict/upload", files=files).status_code == 400
    missing = pd.DataFrame([{"Time": 1}]).to_csv(index=False)
    files = {"file": ("bad.csv", missing, "text/csv")}
    assert client.post("/api/predict/upload", files=files).status_code == 422


@pytest.mark.parametrize("scenario", ["baseline", "schema_shift"])
def test_drift_simulation(client, scenario):
    report = client.get(f"/api/monitor/drift/simulate?scenario={scenario}").json()
    assert report["scenario"] == scenario
    assert report["features_checked"] == len(FEATURE_COLUMNS) + 1
    if scenario == "schema_shift":
        assert report["status"] == "significant"


def test_drift_on_posted_batch(client, benchmark):
    rows = benchmark.drop(columns="Class").head(200).to_dict("records")
    report = client.post("/api/monitor/drift", json=rows).json()
    assert report["rows"] == 200
    assert client.post("/api/monitor/drift", json=rows[:10]).status_code == 400


def test_metrics_track_decisions(client, fraud_row):
    client.post("/api/predict", json=fraud_row)
    snapshot = client.get("/api/metrics").json()
    assert snapshot["transactions_scored"] >= 1
    assert snapshot["requests_by_path"]["/api/predict"] >= 1
