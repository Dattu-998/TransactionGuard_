import json

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from transactionguard.features.engineering import FeatureEngineer
from transactionguard.inference import InferenceError, InferenceService, risk_category


@pytest.fixture
def artifact_dir(tmp_path):
    model_dir = tmp_path / "artifacts"
    model_dir.mkdir()
    train = pd.DataFrame({"F1": [1.0, 2.0, 3.0, 4.0], "F2": [4.0, 3.0, 2.0, 1.0], "Class": [0, 0, 1, 1]})
    engineer = FeatureEngineer().fit(train)
    model = LogisticRegression().fit(engineer.transform(train), train["Class"])
    joblib.dump(engineer, model_dir / "feature_engineer.joblib")
    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "schema.json").write_text(json.dumps({"F1": "float64", "F2": "float64"}))
    (model_dir / "metadata.json").write_text(json.dumps({"model_name": "Test", "threshold": 0.5}))
    return model_dir


def test_batch_prediction_preserves_order_and_ignores_class_and_extra_columns(artifact_dir):
    service = InferenceService(artifact_dir).load()
    source = pd.DataFrame({"id": ["b", "a"], "F1": [1.1, 3.9], "F2": [3.9, 1.1], "Class": [0, 1]})
    result = service.predict(source)
    assert list(result.index) == list(source.index)
    assert len(result) == 2
    assert set(result["Predicted Class"]) <= {"Potential Fraud", "Likely Legitimate"}


def test_validation_reports_missing_and_invalid_values(artifact_dir):
    service = InferenceService(artifact_dir).load()
    _, missing = service.validate(pd.DataFrame({"F1": [1]}))
    assert "Missing required" in missing[0]
    _, invalid = service.validate(pd.DataFrame({"F1": [1], "F2": ["bad"]}))
    assert "non-numeric" in invalid[0]


def test_risk_bands_and_missing_artifacts(tmp_path):
    assert [risk_category(value) for value in (0.1, 0.2, 0.6)] == ["Low", "Medium", "High"]
    with pytest.raises(InferenceError, match="Missing model"):
        InferenceService(tmp_path).load()
