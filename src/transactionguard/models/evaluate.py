from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    if np.sum(y_pred) == 0 and np.sum(y_true) > 0:
        precision = 0.0
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "pr_auc": float(pr_auc), "roc_auc": float(roc_auc), "precision": float(precision),
        "recall": float(recall), "f1": float(f1), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": float(fpr), "fnr": float(fnr), "brier": float(brier_score_loss(y_true, y_prob)),
    }


def pr_curve_points(y_true, y_prob, points: int = 120) -> list[dict]:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    grid = np.linspace(0, 1, points)
    # precision_recall_curve returns recall in decreasing order, so a running max
    # gives the interpolated envelope: the best precision achievable at >= recall r.
    envelope = np.maximum.accumulate(precision)
    sampled = np.interp(grid, recall[::-1], envelope[::-1])
    return [{"recall": float(r), "precision": float(p)} for r, p in zip(grid, sampled, strict=True)]


def roc_curve_points(y_true, y_prob, points: int = 120) -> list[dict]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    grid = np.unique(np.r_[0.0, np.logspace(-5, 0, points - 1)])
    sampled = np.interp(grid, fpr, tpr)
    return [{"fpr": float(f), "tpr": float(t)} for f, t in zip(grid, sampled, strict=True)]


CALIBRATION_EDGES = [0.0, 1e-4, 1e-3, 5e-3, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0000001]


def calibration_points(y_true, y_prob) -> list[dict]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rows = []
    for lo, hi in itertools.pairwise(CALIBRATION_EDGES):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"{lo:g}–{min(hi, 1):g}",
            "predicted": float(y_prob[mask].mean()),
            "observed": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return rows


def score_histogram(y_true, risk_scores, bins: int = 20) -> list[dict]:
    y_true = np.asarray(y_true).astype(int)
    edges = np.linspace(0, 100, bins + 1)
    legit, _ = np.histogram(np.asarray(risk_scores)[y_true == 0], bins=edges)
    fraud, _ = np.histogram(np.asarray(risk_scores)[y_true == 1], bins=edges)
    legit_total, fraud_total = max(int(legit.sum()), 1), max(int(fraud.sum()), 1)
    return [
        {
            "bucket": f"{int(lo)}–{int(hi)}",
            "start": float(lo),
            "legit": int(lc),
            "fraud": int(fc),
            "legit_share": float(lc / legit_total),
            "fraud_share": float(fc / fraud_total),
        }
        for lo, hi, lc, fc in zip(edges[:-1], edges[1:], legit, fraud, strict=True)
    ]
