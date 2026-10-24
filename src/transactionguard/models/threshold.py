import numpy as np

from transactionguard.config import config


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, fn_cost: float = config.FN_COST, fp_cost: float = config.FP_COST) -> tuple[float, dict]:
    # Try dynamic thresholds based on actual probability percentiles to cover real range
    if len(y_prob) > 100:
        thresholds = np.percentile(y_prob, np.linspace(1, 99, 99))
    else:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    min_cost = float('inf')
    best_metrics = {}

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        cost = (fn * fn_cost) + (fp * fp_cost)

        if cost < min_cost:
            min_cost = cost
            best_threshold = th
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            best_metrics = {
                "threshold": float(th),
                "cost": float(cost),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            }

    return float(best_threshold), best_metrics
