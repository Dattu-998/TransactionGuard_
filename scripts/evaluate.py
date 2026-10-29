"""Re-score the held-out test split with every registered version and print a leaderboard."""

import json

import pandas as pd

from transactionguard.config import config
from transactionguard.models.evaluate import evaluate_model
from transactionguard.models.registry import ModelRegistry
from transactionguard.scoring import positive_probabilities, validate_frame
from transactionguard.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    test_path = config.PROCESSED_DATA_DIR / "test_data.csv"
    if not test_path.exists():
        logger.error("Test data not found; run scripts/train.py first.")
        return
    test = pd.read_csv(test_path)
    features, labels = validate_frame(test), test[config.TARGET_COL]
    registry = ModelRegistry(config.ARTIFACT_DIR).load()
    leaderboard = {}
    for version in registry.versions:
        model = registry.get(version)
        probabilities = positive_probabilities(model.estimator, features)
        metrics = evaluate_model(labels, probabilities, threshold=model.policy.review_threshold)
        leaderboard[version] = metrics
        flag = "*" if version == registry.champion else " "
        logger.info("%s%s PR-AUC %.4f  ROC-AUC %.4f  recall %.3f  precision %.3f  brier %.5f",
                    flag, version, metrics["pr_auc"], metrics["roc_auc"], metrics["recall"],
                    metrics["precision"], metrics["brier"])
    out = config.REPORT_DIR / "leaderboard.json"
    out.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
    logger.info("Leaderboard written to %s", out)


if __name__ == "__main__":
    main()
