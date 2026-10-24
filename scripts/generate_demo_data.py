from pathlib import Path

import numpy as np
import pandas as pd

from transactionguard.config import config
from transactionguard.utils.logging import get_logger

logger = get_logger(__name__)

def generate_demo_dataset(output_path: Path, num_samples: int = 284807):
    logger.info(f"Generating synthetic demo dataset with {num_samples} samples...")
    np.random.seed(42)
    time_col = np.sort(np.random.uniform(0, 172792, num_samples))
    v_cols = {f"V{i}": np.random.normal(0, 1, num_samples) for i in range(1, 29)}
    amount_col = np.random.lognormal(mean=3, sigma=1.5, size=num_samples)
    fraud_prob = 0.00172
    class_col = np.random.choice([0, 1], size=num_samples, p=[1 - fraud_prob, fraud_prob])
    v_cols["V1"][class_col == 1] += 5.0
    v_cols["V2"][class_col == 1] -= 5.0
    df_data = {"Time": time_col}
    df_data.update(v_cols)
    df_data["Amount"] = amount_col
    df_data["Class"] = class_col
    df = pd.DataFrame(df_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Generated synthetic demo data at {output_path}")

if __name__ == "__main__":
    generate_demo_dataset(config.RAW_DATA_PATH)
