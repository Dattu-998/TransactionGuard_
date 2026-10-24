import numpy as np
import pandas as pd

from transactionguard.config import config
from transactionguard.data.split import split_data


def test_split_data():
    num_samples = 1000
    np.random.seed(42)
    df = pd.DataFrame({'Feature1': np.random.randn(num_samples), 'Class': np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])})
    orig_fraud_ratio = df['Class'].mean()
    train, val, test = split_data(df)
    assert len(train) + len(val) + len(test) == num_samples
    test_expected = int(num_samples * config.TEST_SIZE)
    val_expected = int(num_samples * config.VAL_SIZE)
    assert abs(len(test) - test_expected) <= 1
    assert abs(len(val) - val_expected) <= 1
    assert abs(train['Class'].mean() - orig_fraud_ratio) < 0.05
    assert abs(val['Class'].mean() - orig_fraud_ratio) < 0.05
    assert abs(test['Class'].mean() - orig_fraud_ratio) < 0.05
    assert len(set(train.index).intersection(set(val.index))) == 0
    assert len(set(train.index).intersection(set(test.index))) == 0
    assert len(set(val.index).intersection(set(test.index))) == 0
