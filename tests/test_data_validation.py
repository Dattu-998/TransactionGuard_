import numpy as np
import pandas as pd
import pytest

from transactionguard.data.validate import validate_dataset, validate_features


def test_validate_dataset_valid():
    df = pd.DataFrame({'Feature1': [1.0, 2.0], 'Class': [0, 1]})
    assert validate_dataset(df) is True

def test_validate_dataset_empty():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_dataset(df)

def test_validate_dataset_missing_target():
    df = pd.DataFrame({'Feature1': [1.0, 2.0]})
    with pytest.raises(ValueError, match="Target column 'Class' not found"):
        validate_dataset(df)

def test_validate_dataset_invalid_classes():
    df = pd.DataFrame({'Feature1': [1.0, 2.0], 'Class': [0, 2]})
    with pytest.raises(ValueError, match="Target column must only contain 0 and 1"):
        validate_dataset(df)

def test_validate_dataset_inf_values():
    df = pd.DataFrame({'Feature1': [1.0, np.inf], 'Class': [0, 1]})
    with pytest.raises(ValueError, match="Dataset contains infinite values"):
        validate_dataset(df)

def test_validate_features_valid():
    df = pd.DataFrame({'V1': [1.0], 'V2': [2.0]})
    assert validate_features(df, ['V1', 'V2']) is True

def test_validate_features_missing():
    df = pd.DataFrame({'V1': [1.0]})
    with pytest.raises(ValueError, match="Missing expected features"):
        validate_features(df, ['V1', 'V2'])
