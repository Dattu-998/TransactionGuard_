"""Train every model version and publish the registry to ``artifacts/``.

    python scripts/train.py                 # train v1, v2, v3 on data/raw/creditcard.csv
    python scripts/train.py --regenerate    # rebuild the synthetic benchmark first
    python scripts/train.py --versions v3   # retrain a subset
"""

import argparse

from transactionguard.config import config
from transactionguard.data.load import load_data
from transactionguard.data.synthetic import write_benchmark
from transactionguard.training import dataset_info, train_registry
from transactionguard.utils.logging import get_logger

logger = get_logger(__name__)


def _log_to_mlflow(index: dict) -> None:
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(f"sqlite:///{(config.BASE_DIR / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("TransactionGuard-Fraud-Detection")
    for entry in index["versions"]:
        with mlflow.start_run(run_name=f"{entry['version']} {entry['name']}"):
            mlflow.log_params({"version": entry["version"], "stage": entry["stage"],
                               "review_threshold": entry["thresholds"]["review"]})
            mlflow.log_metrics({k: v for k, v in entry["metrics"].items() if v is not None})
            mlflow.log_artifact(str(config.ARTIFACT_DIR / "models" / entry["version"] / "card.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true", help="regenerate synthetic data")
    parser.add_argument("--versions", nargs="*", help="subset of versions to train")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()

    path = config.RAW_DATA_PATH
    if args.regenerate or not path.exists():
        logger.info("Generating synthetic benchmark at %s", path)
        write_benchmark(path)

    index = train_registry(load_data(path), config.ARTIFACT_DIR, dataset_info(path), args.versions,
                           processed_dir=config.PROCESSED_DATA_DIR)
    if not args.no_mlflow:
        _log_to_mlflow(index)
    for entry in index["versions"]:
        m = entry["metrics"]
        logger.info("%-3s %-38s %-10s PR-AUC %.4f  recall %.3f  precision %.3f",
                    entry["version"], entry["name"], entry["stage"], m["pr_auc"],
                    m["recall"], m["precision"])


if __name__ == "__main__":
    main()
