import pytest

from transactionguard.data.synthetic import generate_transactions
from transactionguard.training import train_registry


@pytest.fixture(scope="session")
def benchmark():
    return generate_transactions(n_samples=12_000, fraud_rate=0.02, seed=7)


@pytest.fixture(scope="session")
def registry_dir(tmp_path_factory, benchmark):
    """A small but complete registry (all three versions) trained once per test session."""
    directory = tmp_path_factory.mktemp("artifacts")
    train_registry(benchmark, directory, {"source": "synthetic", "description": "test"})
    return directory
