"""Cost-driven decision thresholds.

Every distinct predicted probability is evaluated as a candidate threshold. The
earlier implementation only tried probability percentiles, which meant it could
never flag fewer than 1% of transactions and over-flagged hundreds of customers.
"""

from __future__ import annotations

import numpy as np

from transactionguard.config import config


def _cumulative_counts(y_true: np.ndarray, y_prob: np.ndarray):
    """For each distinct score s (descending), counts of positives/negatives with p >= s."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(-y_prob, kind="stable")
    sorted_prob = y_prob[order]
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true)
    fp = np.cumsum(1 - sorted_true)
    # Keep the last index of each tie group so "p >= threshold" includes every tie.
    last_of_group = np.r_[np.diff(sorted_prob) != 0, True]
    return sorted_prob[last_of_group], tp[last_of_group], fp[last_of_group], int(y_true.sum()), len(y_true)


def _metrics(tp: int, fp: int, positives: int, total: int, threshold: float, cost: float) -> dict:
    fn = positives - tp
    tn = total - positives - fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold), "cost": float(cost),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = config.FN_COST,
    fp_cost: float = config.FP_COST,
) -> tuple[float, dict]:
    """Threshold that minimises ``fn * fn_cost + fp * fp_cost`` on the given labels."""
    thresholds, tp, fp, positives, total = _cumulative_counts(y_true, y_prob)
    costs = (positives - tp) * fn_cost + fp * fp_cost
    best = int(np.argmin(costs))
    flag_nothing_cost = positives * fn_cost
    if flag_nothing_cost < costs[best]:
        threshold = float(np.nextafter(thresholds[0], np.inf))
        return threshold, _metrics(0, 0, positives, total, threshold, flag_nothing_cost)
    threshold = float(thresholds[best])
    return threshold, _metrics(tp[best], fp[best], positives, total, threshold, costs[best])


def precision_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, min_precision: float, floor: float
) -> float:
    """Lowest threshold (>= floor) whose precision reaches ``min_precision``.

    Used for the automatic-block band: above it the model is confident enough to
    decline outright instead of sending the transaction to an analyst.
    """
    thresholds, tp, fp, _, _ = _cumulative_counts(y_true, y_prob)
    precision = tp / np.maximum(tp + fp, 1)
    eligible = (precision >= min_precision) & (thresholds >= floor) & (tp >= 3)
    if not eligible.any():
        return float(max(floor, np.quantile(y_prob, 0.9995)))
    return float(thresholds[eligible].min())


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = config.FN_COST,
    fp_cost: float = config.FP_COST,
    points: int = 90,
) -> list[dict]:
    """Precision / recall / cost across a log-spaced threshold grid, for the UI explorer."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    positives, total = int(y_true.sum()), len(y_true)
    grid = np.unique(np.r_[np.logspace(-5, np.log10(0.995), points)])
    rows = []
    for th in grid:
        flagged = y_prob >= th
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        cost = (positives - tp) * fn_cost + fp * fp_cost
        m = _metrics(tp, fp, positives, total, th, cost)
        m["flag_rate"] = float(flagged.mean())
        rows.append(m)
    return rows
