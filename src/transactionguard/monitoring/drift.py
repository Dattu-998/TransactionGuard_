from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_BINS = 10
PSI_EPSILON = 1e-4
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25


def calculate_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, numerical_features: list[str]) -> dict:
    """Two-sample Kolmogorov-Smirnov drift test per feature (offline reporting)."""
    drift_results = {}
    total_drifted = 0
    for feature in numerical_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        ref_data = reference_df[feature].dropna()
        curr_data = current_df[feature].dropna()
        if len(ref_data) == 0 or len(curr_data) == 0:
            continue
        stat, p_value = ks_2samp(ref_data, curr_data)
        is_drifted = bool(p_value < 0.05)
        if is_drifted:
            total_drifted += 1
        drift_results[feature] = {"statistic": float(stat), "p_value": float(p_value), "is_drifted": is_drifted}
    return {
        "features_checked": len(numerical_features),
        "features_drifted": total_drifted,
        "drift_fraction": total_drifted / len(numerical_features) if len(numerical_features) > 0 else 0.0,
        "feature_details": drift_results
    }


def build_reference(values: np.ndarray, bins: int = PSI_BINS) -> dict:
    """Quantile bin edges and expected proportions for one feature."""
    values = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(values, bins=edges)
    return {
        "edges": [float(e) if np.isfinite(e) else None for e in edges],
        "expected": (counts / counts.sum()).tolist(),
        "mean": float(values.mean()),
        "std": float(values.std() or 1.0),
    }


def _edges(reference: dict) -> np.ndarray:
    raw = reference["edges"]
    return np.array([-np.inf, *raw[1:-1], np.inf], dtype=float)


def population_stability_index(reference: dict, current: np.ndarray) -> tuple[float, list[float]]:
    counts, _ = np.histogram(np.asarray(current, dtype=float), bins=_edges(reference))
    actual = counts / max(counts.sum(), 1)
    expected = np.asarray(reference["expected"], dtype=float)
    a = np.clip(actual, PSI_EPSILON, None)
    e = np.clip(expected, PSI_EPSILON, None)
    return float(np.sum((a - e) * np.log(a / e))), actual.tolist()


def psi_status(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def drift_report(references: dict[str, dict], current: pd.DataFrame) -> dict:
    """PSI and standardised mean shift for every referenced column present in ``current``."""
    features = []
    for name, reference in references.items():
        if name not in current.columns:
            continue
        values = pd.to_numeric(current[name], errors="coerce").dropna().to_numpy()
        if not len(values):
            continue
        psi, actual = population_stability_index(reference, values)
        features.append({
            "feature": name,
            "psi": psi,
            "status": psi_status(psi),
            "mean_shift": float((values.mean() - reference["mean"]) / reference["std"]),
            "expected": reference["expected"],
            "actual": actual,
        })
    features.sort(key=lambda item: item["psi"], reverse=True)
    counts = {status: sum(f["status"] == status for f in features)
              for status in ("stable", "moderate", "significant")}
    overall = "significant" if counts["significant"] else "moderate" if counts["moderate"] else "stable"
    return {
        "rows": len(current),
        "features_checked": len(features),
        "status": overall,
        "counts": counts,
        "max_psi": features[0]["psi"] if features else 0.0,
        "features": features,
    }
