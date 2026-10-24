import json
import time

import joblib
import mlflow
import mlflow.sklearn

from transactionguard.config import config
from transactionguard.data.load import load_data
from transactionguard.data.split import split_data
from transactionguard.data.validate import validate_dataset
from transactionguard.features.engineering import FeatureEngineer
from transactionguard.models.evaluate import evaluate_model
from transactionguard.models.threshold import optimize_threshold
from transactionguard.models.train import (
    train_lightgbm,
    train_logistic_regression,
    train_random_forest,
)
from transactionguard.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Starting training pipeline...")
    mlflow.set_tracking_uri(
    f"sqlite:///{(config.BASE_DIR / 'mlflow.db').as_posix()}"
)
    mlflow.set_experiment("TransactionGuard-Fraud-Detection")

    try:
        df = load_data(config.RAW_DATA_PATH)
    except FileNotFoundError:
        logger.error("Data not found.")
        return

    validate_dataset(df)
    train_df, val_df, test_df = split_data(df)
    test_df.to_csv(config.PROCESSED_DATA_DIR / "test_data.csv", index=False)

    # Save train_df as reference for drift monitoring
    train_df.to_csv(config.REFERENCE_DATA_DIR / "reference.csv", index=False)

    fe = FeatureEngineer()
    X_train = fe.fit_transform(train_df)
    y_train = train_df[config.TARGET_COL]
    X_val = fe.transform(val_df)
    y_val = val_df[config.TARGET_COL]

    joblib.dump(fe, config.ARTIFACT_DIR / "feature_engineer.joblib")

    models = {
        "Logistic Regression": train_logistic_regression,
        "Random Forest": train_random_forest,
        "LightGBM": train_lightgbm
    }

    best_model_name = None
    best_pr_auc = -1

    for model_name, train_func in models.items():
        with mlflow.start_run(run_name=model_name):
            logger.info(f"--- Evaluating {model_name} ---")

            start_time = time.time()
            model = train_func(X_train, y_train)
            train_duration = time.time() - start_time

            if hasattr(model, "predict_proba"):
                y_prob_val = model.predict_proba(X_val)[:, 1]
            else:
                y_prob_val = model.predict(X_val)

            metrics_05 = evaluate_model(y_val, y_prob_val, threshold=0.5)
            best_th, opt_metrics = optimize_threshold(y_val, y_prob_val)

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("random_seed", config.RANDOM_STATE)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            mlflow.log_param("fn_cost", config.FN_COST)
            mlflow.log_param("fp_cost", config.FP_COST)

            mlflow.log_metric("val_pr_auc", metrics_05["pr_auc"])
            mlflow.log_metric("val_roc_auc", metrics_05["roc_auc"])
            mlflow.log_metric("val_precision", opt_metrics.get("precision", 0))
            mlflow.log_metric("val_recall", opt_metrics.get("recall", 0))
            mlflow.log_metric("val_f1", opt_metrics.get("f1", 0))
            mlflow.log_metric("opt_threshold", best_th)
            mlflow.log_metric("train_duration_sec", train_duration)
            mlflow.log_metric("tn", opt_metrics.get("tn", 0))
            mlflow.log_metric("fp", opt_metrics.get("fp", 0))
            mlflow.log_metric("fn", opt_metrics.get("fn", 0))
            mlflow.log_metric("tp", opt_metrics.get("tp", 0))
            mlflow.log_metric("cost", opt_metrics.get("cost", 0))

            # Log Model Artifact natively using joblib instead of sklearn to bypass skops trusted types errors entirely
            # AND doing it without the silent failure catch
            temp_path = config.ARTIFACT_DIR / f"{model_name.replace(' ', '_')}.joblib"
            joblib.dump(model, temp_path)
            mlflow.log_artifact(str(temp_path), artifact_path="model")

            logger.info(f"{model_name} Val PR-AUC: {metrics_05['pr_auc']:.4f}")
            logger.info(f"{model_name} Opt Threshold: {best_th:.4f}")

            if metrics_05["pr_auc"] > best_pr_auc:
                best_pr_auc = metrics_05["pr_auc"]
                best_model_name = model_name
                joblib.dump(model, config.ARTIFACT_DIR / "model.joblib")
                metadata = {
                    "model_name": model_name,
                    "threshold": best_th,
                    "pr_auc": metrics_05["pr_auc"],
                    "cost_metrics": opt_metrics,
                    "feature_names": fe.feature_names_in_,
                    "version": "v1.0"
                }
                with open(config.ARTIFACT_DIR / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=4)
                schema = {col: str(train_df[col].dtype) for col in train_df.columns if col != config.TARGET_COL}
                with open(config.ARTIFACT_DIR / "schema.json", "w") as f:
                    json.dump(schema, f, indent=4)

                # Log schemas to mlflow too
                mlflow.log_artifact(str(config.ARTIFACT_DIR / "metadata.json"), artifact_path="metadata")
                mlflow.log_artifact(str(config.ARTIFACT_DIR / "schema.json"), artifact_path="metadata")

    logger.info(f"Selected best model: {best_model_name}")

if __name__ == "__main__":
    main()
