import numpy as np
import pandas as pd

from transactionguard.monitoring.drift import calculate_drift


def test_calculate_drift_no_drift():
    np.random.seed(42)
    ref = pd.DataFrame({'F1': np.random.normal(0, 1, 1000)})
    curr = pd.DataFrame({'F1': np.random.normal(0, 1, 1000)})
    summary = calculate_drift(ref, curr, ['F1'])
    assert summary['features_checked'] == 1
    assert summary['features_drifted'] == 0
    assert summary['feature_details']['F1']['is_drifted'] is False

def test_calculate_drift_with_drift():
    np.random.seed(42)
    ref = pd.DataFrame({'F1': np.random.normal(0, 1, 1000)})
    curr = pd.DataFrame({'F1': np.random.normal(5, 1, 1000)})
    summary = calculate_drift(ref, curr, ['F1'])
    assert summary['features_checked'] == 1
    assert summary['features_drifted'] == 1
    assert summary['feature_details']['F1']['is_drifted'] is True

def test_calculate_drift_missing_feature():
    ref = pd.DataFrame({'F1': [1, 2, 3]})
    curr = pd.DataFrame({'F2': [1, 2, 3]})
    summary = calculate_drift(ref, curr, ['F1'])
    assert summary['features_checked'] == 1
    assert summary['features_drifted'] == 0
    assert 'F1' not in summary['feature_details']
