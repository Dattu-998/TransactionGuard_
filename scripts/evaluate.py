import json

import joblib
import pandas as pd

from transactionguard.config import config
from transactionguard.models.evaluate import evaluate_model, plot_confusion_matrix
from transactionguard.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    test_data_path = config.PROCESSED_DATA_DIR / "test_data.csv"
    if not test_data_path.exists():
        logger.error("Test data not found.")
        return
    df_test = pd.read_csv(test_data_path)
    try:
        model = joblib.load(config.ARTIFACT_DIR / "model.joblib")
        fe = joblib.load(config.ARTIFACT_DIR / "feature_engineer.joblib")
        with open(config.ARTIFACT_DIR / "metadata.json", "r") as f:
            metadata = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Artifact not found: {e}")
        return

    threshold = metadata.get("threshold", 0.5)
    X_test = fe.transform(df_test)
    y_test = df_test[config.TARGET_COL]

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)

    metrics = evaluate_model(y_test, y_prob, threshold=threshold)
    with open(config.REPORT_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    y_pred = (y_prob >= threshold).astype(int)
    plot_confusion_matrix(y_test, y_pred, config.REPORT_DIR / "confusion_matrix.png")

    logger.info(f"Test PR-AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")

if __name__ == "__main__":
    main()
