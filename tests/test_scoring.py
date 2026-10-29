import numpy as np
import pandas as pd
import pytest

from transactionguard.features.engineering import FEATURE_COLUMNS
from transactionguard.monitoring.drift import (
    build_reference,
    drift_report,
    population_stability_index,
)
from transactionguard.scoring import DecisionPolicy, ScoringError, risk_level, validate_frame


def test_risk_score_anchors_and_monotonicity():
    policy = DecisionPolicy(review_threshold=0.01, block_threshold=0.4)
    scores = policy.risk_scores(np.array([1e-7, 0.01, 0.4, 1.0]))
    assert scores[0] == 0
    assert scores[1] == pytest.approx(50)
    assert scores[2] == pytest.approx(80)
    assert scores[3] == pytest.approx(100)
    grid = policy.risk_scores(np.logspace(-6, 0, 50))
    assert np.all(np.diff(grid) >= 0)


def test_decision_bands():
    policy = DecisionPolicy(review_threshold=0.01, block_threshold=0.4)
    assert list(policy.decisions(np.array([0.001, 0.01, 0.2, 0.4, 0.9]))) == [
        "approve", "review", "review", "block", "block"]
    assert [risk_level(s) for s in (10, 35, 60, 90)] == ["low", "elevated", "high", "critical"]


def _row(**overrides):
    row = {name: 0.0 for name in FEATURE_COLUMNS}
    row.update(overrides)
    return row


def test_validate_frame_reports_problems():
    with pytest.raises(ScoringError, match="Missing required"):
        validate_frame(pd.DataFrame([{"Time": 1.0}]))
    with pytest.raises(ScoringError, match="non-numeric"):
        validate_frame(pd.DataFrame([_row(V3="abc")]))
    with pytest.raises(ScoringError, match="infinite"):
        validate_frame(pd.DataFrame([_row(Amount=np.inf)]))
    with pytest.raises(ScoringError, match="no transaction rows"):
        validate_frame(pd.DataFrame(columns=FEATURE_COLUMNS))
    frame = validate_frame(pd.DataFrame([_row(V1="1.5", extra="kept-out")]))
    assert list(frame.columns) == FEATURE_COLUMNS and frame["V1"].iloc[0] == 1.5


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    reference = build_reference(rng.normal(0, 1, 5000))
    stable, _ = population_stability_index(reference, rng.normal(0, 1, 2000))
    shifted, _ = population_stability_index(reference, rng.normal(1.0, 1, 2000))
    assert stable < 0.1
    assert shifted > 0.25


def test_drift_report_summary():
    rng = np.random.default_rng(1)
    references = {"a": build_reference(rng.normal(0, 1, 3000)),
                  "b": build_reference(rng.normal(0, 1, 3000))}
    current = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(2, 1, 1000)})
    report = drift_report(references, current)
    assert report["status"] == "significant"
    assert report["features"][0]["feature"] == "b"
    assert report["counts"] == {"stable": 1, "moderate": 0, "significant": 1}


def test_pr_curve_envelope_decreases_with_recall():
    from transactionguard.models.evaluate import pr_curve_points

    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(2000), np.ones(40)].astype(int)
    p = np.r_[rng.beta(1, 8, 2000), rng.beta(4, 2, 40)]
    precision = [point["precision"] for point in pr_curve_points(y, p)]
    assert np.all(np.diff(precision) <= 1e-12)
    assert precision[0] == 1.0 and precision[-1] < 0.5
