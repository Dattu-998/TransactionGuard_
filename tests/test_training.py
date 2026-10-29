import json

import numpy as np

from transactionguard.data.synthetic import generate_transactions
from transactionguard.features.engineering import FEATURE_COLUMNS, TransactionFeatureBuilder
from transactionguard.models.registry import ModelRegistry


def test_synthetic_benchmark_schema_and_prevalence():
    frame = generate_transactions(n_samples=5_000, fraud_rate=0.01, seed=1)
    assert list(frame.columns) == [*FEATURE_COLUMNS, "Class"]
    assert frame["Class"].sum() == 50
    assert frame["Time"].is_monotonic_increasing
    assert np.isfinite(frame[FEATURE_COLUMNS].to_numpy()).all()


def test_feature_builder_adds_behavioral_features(benchmark):
    builder = TransactionFeatureBuilder().fit(benchmark[FEATURE_COLUMNS])
    out = builder.transform(benchmark[FEATURE_COLUMNS].head(10))
    assert list(out.columns) == list(builder.get_feature_names_out())
    assert {"log_amount", "hour_sin", "is_night", "component_norm"} <= set(out.columns)
    assert out["is_night"].isin([0.0, 1.0]).all()


def test_registry_has_every_version_with_a_champion(registry_dir):
    index = json.loads((registry_dir / "registry.json").read_text())
    versions = {entry["version"]: entry for entry in index["versions"]}
    assert set(versions) == {"v1", "v2", "v3"}
    assert index["champion"] in versions
    assert versions[index["champion"]]["stage"] == "champion"
    assert sorted(e["stage"] for e in versions.values()) == ["archived", "challenger", "champion"]


def test_model_card_contents(registry_dir):
    card = json.loads((registry_dir / "models" / "v3" / "card.json").read_text())
    thresholds = card["thresholds"]
    assert 0 < thresholds["review"] <= thresholds["block"] < 1
    assert {"pr", "roc", "calibration", "threshold_sweep", "score_histogram"} <= set(card["curves"])
    assert len(card["importance"]) == len(FEATURE_COLUMNS)
    assert abs(sum(card["decision_mix"].values()) - 1) < 1e-9
    assert card["latency"]["p50_ms"] > 0


def test_registry_scores_and_explains(registry_dir, benchmark):
    registry = ModelRegistry(registry_dir).load()
    fraud = benchmark[benchmark["Class"] == 1].head(5)
    results, reasons, elapsed = registry.score(fraud, explain_rows="all")
    assert set(results.columns) == {"fraud_probability", "risk_score", "risk_level", "decision"}
    assert results["risk_score"].between(0, 100).all()
    assert all(reasons) and elapsed >= 0
    assert reasons[0][0]["feature"] in FEATURE_COLUMNS
    # The champion should rank fraud above typical legitimate traffic.
    legit, _, _ = registry.score(benchmark[benchmark["Class"] == 0].head(200), explain_rows="none")
    assert results["fraud_probability"].median() > legit["fraud_probability"].median()


def test_registry_falls_back_when_champion_cannot_load(registry_dir, tmp_path):
    import shutil

    copy = tmp_path / "artifacts"
    shutil.copytree(registry_dir, copy)
    champion = json.loads((copy / "registry.json").read_text())["champion"]
    (copy / "models" / champion / "model.joblib").write_bytes(b"not a pickle")
    registry = ModelRegistry(copy).load()
    assert champion in registry.load_errors
    assert champion not in registry.versions
    assert registry.champion in registry.versions
    assert [e["version"] for e in registry.index["versions"]] == registry.versions
