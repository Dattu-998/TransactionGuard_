"""Synthetic card-transaction benchmark that mirrors the Kaggle credit-card schema.

The original demo data shifted two features by five standard deviations, which made
every model score a perfect 1.0 and hid any real difference between approaches. This
generator produces a deliberately harder problem so that model versions can be
compared honestly:

* legitimate customers come from several correlated, heavy-tailed segments with a
  daily activity cycle;
* fraud is a mixture of distinct attack patterns, one of which is non-linear and one
  of which is nearly indistinguishable from normal traffic;
* a small share of legitimate transactions look suspicious (travel, large purchases),
  which creates realistic false-positive pressure.

The columns are ``Time``, ``V1``-``V28``, ``Amount`` and ``Class`` so the same pipeline
can be trained on the real Kaggle file without code changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_COMPONENTS = 28
SECONDS_PER_DAY = 86_400
DAYS = 2

# Relative activity by hour of day (index 0 = midnight).
LEGIT_HOURLY = np.array(
    [0.9, 0.5, 0.35, 0.3, 0.35, 0.6, 1.2, 2.2, 3.4, 4.2, 4.6, 4.8,
     5.0, 4.9, 4.7, 4.6, 4.7, 4.9, 5.0, 4.6, 3.9, 3.1, 2.2, 1.5]
)
FRAUD_HOURLY = np.array(
    [3.2, 3.6, 3.8, 3.6, 3.0, 2.2, 1.6, 1.4, 1.5, 1.7, 1.8, 1.9,
     2.0, 2.0, 1.9, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.6, 2.8, 3.0]
)

FRAUD_PATTERNS = {
    # name: (share of fraud, feature shifts, noise scale)
    "card_testing": (0.30, {"V14": -2.4, "V12": -1.7, "V4": 1.7, "V10": -1.1}, 0.9),
    "account_takeover": (0.36, {"V14": -3.8, "V17": -3.0, "V10": -2.4, "V11": 2.0,
                                "V12": -2.1, "V4": 2.2, "V3": -1.4}, 1.4),
    "merchant_compromise": (0.22, {}, 0.95),  # non-linear ring in the V3/V7 plane
    "stealth": (0.12, {"V14": -1.1, "V4": 0.8, "V17": -0.7}, 1.0),
}


def _hourly_times(rng: np.random.Generator, n: int, weights: np.ndarray) -> np.ndarray:
    hours = rng.choice(24, size=n, p=weights / weights.sum())
    days = rng.integers(0, DAYS, size=n)
    return days * SECONDS_PER_DAY + hours * 3600 + rng.uniform(0, 3600, size=n)


def _heavy_tailed(rng: np.random.Generator, n: int, mixing: np.ndarray, dof: float = 9.0):
    """Correlated multivariate Student-t draws, standardised to unit variance."""
    z = rng.standard_normal((n, N_COMPONENTS)) @ mixing.T
    scale = np.sqrt(rng.chisquare(dof, size=(n, 1)) / dof)
    return z / scale / np.sqrt(dof / (dof - 2))


def generate_transactions(
    n_samples: int = 284_807,
    fraud_rate: float = 0.0021,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, round(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    # Shared correlation structure (rows normalised so each component has unit variance).
    mixing = np.eye(N_COMPONENTS) + rng.normal(0, 0.18, (N_COMPONENTS, N_COMPONENTS))
    mixing /= np.linalg.norm(mixing, axis=1, keepdims=True)

    # --- legitimate traffic: three customer segments -----------------------------
    segment = rng.choice(3, size=n_legit, p=[0.62, 0.28, 0.10])
    segment_means = rng.normal(0, 0.45, (3, N_COMPONENTS))
    legit_v = _heavy_tailed(rng, n_legit, mixing) + segment_means[segment]
    legit_amount = np.where(
        segment == 0,
        rng.lognormal(3.1, 1.05, n_legit),
        np.where(segment == 1, rng.lognormal(4.0, 0.9, n_legit), rng.lognormal(2.2, 0.8, n_legit)),
    )
    # Legitimate-but-suspicious behaviour (travel, big-ticket purchases).
    unusual = rng.random(n_legit) < 0.015
    legit_v[unusual, 13] -= rng.normal(1.8, 0.6, unusual.sum())  # V14
    legit_v[unusual, 3] += rng.normal(1.0, 0.5, unusual.sum())  # V4
    legit_amount[unusual] *= rng.lognormal(0.8, 0.4, unusual.sum())
    legit_time = _hourly_times(rng, n_legit, LEGIT_HOURLY)

    # --- fraud: mixture of attack patterns ----------------------------------------
    names = list(FRAUD_PATTERNS)
    shares = np.array([FRAUD_PATTERNS[k][0] for k in names])
    pattern = rng.choice(len(names), size=n_fraud, p=shares / shares.sum())
    fraud_v = _heavy_tailed(rng, n_fraud, mixing, dof=7.0)
    fraud_amount = np.empty(n_fraud)
    for idx, name in enumerate(names):
        rows = pattern == idx
        count = int(rows.sum())
        if not count:
            continue
        _, shifts, noise = FRAUD_PATTERNS[name]
        fraud_v[rows] *= noise
        for feature, shift in shifts.items():
            column = int(feature[1:]) - 1
            fraud_v[rows, column] += shift * rng.normal(1.0, 0.35, count)
        if name == "card_testing":
            fraud_amount[rows] = rng.uniform(0.5, 6.0, count)
        elif name == "account_takeover":
            fraud_amount[rows] = rng.lognormal(5.6, 0.75, count)
        elif name == "merchant_compromise":
            angle = rng.uniform(0, 2 * np.pi, count)
            radius = rng.normal(3.3, 0.25, count)
            fraud_v[rows, 2] = radius * np.cos(angle)  # V3
            fraud_v[rows, 6] = radius * np.sin(angle)  # V7
            fraud_amount[rows] = rng.lognormal(3.6, 0.9, count)
        else:
            fraud_amount[rows] = rng.lognormal(3.4, 1.0, count)
    fraud_time = _hourly_times(rng, n_fraud, FRAUD_HOURLY)

    v = np.vstack([legit_v, fraud_v])
    frame = pd.DataFrame(v, columns=[f"V{i}" for i in range(1, N_COMPONENTS + 1)])
    frame.insert(0, "Time", np.concatenate([legit_time, fraud_time]))
    frame["Amount"] = np.clip(np.concatenate([legit_amount, fraud_amount]), 0.0, 25_691.16)
    frame["Class"] = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    frame = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    numeric = frame.columns.drop("Class")
    frame[numeric] = frame[numeric].round(6)
    frame["Amount"] = frame["Amount"].round(2)
    return frame


SOURCE_INFO = {
    "source": "synthetic",
    "description": (
        "TransactionGuard synthetic benchmark: Kaggle-schema transactions with four "
        "fraud patterns (card testing, account takeover, merchant compromise, stealth)"
    ),
}


def write_benchmark(path, **kwargs) -> pd.DataFrame:
    """Generate the benchmark to ``path`` plus a ``.source.json`` provenance sidecar."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_transactions(**kwargs)
    frame.to_csv(path, index=False)
    path.with_suffix(".source.json").write_text(json.dumps(SOURCE_INFO), encoding="utf-8")
    return frame
