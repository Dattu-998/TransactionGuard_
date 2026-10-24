import json

import pandas as pd


def generate_drift_report(drift_summary: dict, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "drift_summary.json", 'w') as f:
        json.dump(drift_summary, f, indent=4)
    if drift_summary.get("feature_details"):
        df_details = pd.DataFrame.from_dict(drift_summary["feature_details"], orient='index')
        df_details.index.name = 'feature'
        df_details.to_csv(report_dir / "drift_features.csv")
