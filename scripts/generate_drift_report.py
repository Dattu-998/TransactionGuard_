import pandas as pd

from transactionguard.config import config
from transactionguard.monitoring.drift import calculate_drift
from transactionguard.monitoring.report import generate_drift_report
from transactionguard.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    try:
        df = pd.read_csv(config.RAW_DATA_PATH)
    except FileNotFoundError:
        logger.error("Raw data not found.")
        return
    df_sorted = df.sort_values(by="Time")
    midpoint = len(df_sorted) // 2
    ref_df = df_sorted.iloc[:midpoint]
    curr_df = df_sorted.iloc[midpoint:]
    num_features = [col for col in ref_df.columns if col != config.TARGET_COL and pd.api.types.is_numeric_dtype(ref_df[col])]
    summary = calculate_drift(ref_df, curr_df, num_features)
    generate_drift_report(summary, config.REPORT_DIR)

if __name__ == "__main__":
    main()
