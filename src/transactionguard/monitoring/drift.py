import pandas as pd
from scipy.stats import ks_2samp


def calculate_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, numerical_features: list[str]) -> dict:
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
