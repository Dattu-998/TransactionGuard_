import numpy as np

from transactionguard.models.threshold import optimize_threshold


def test_optimize_threshold_perfect_predictor():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    th, metrics = optimize_threshold(y_true, y_prob, fn_cost=100, fp_cost=10)
    assert metrics["cost"] == 0.0
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert th > 0.2 and th <= 0.8

def test_optimize_threshold_imbalanced_cost():
    y_true = np.array([1, 0])
    y_prob = np.array([0.3, 0.4])
    th, metrics = optimize_threshold(y_true, y_prob, fn_cost=1000, fp_cost=10)
    assert th <= 0.3
    assert metrics["fn"] == 0
    assert metrics["fp"] == 1
    assert metrics["cost"] == 10
