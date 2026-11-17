import numpy as np
import pandas as pd
import pytest

from cdmv.pipeline import build
from cdmv.warehouse import RAW_COLUMNS, connect, load_raw


def synthetic_raw(n=500, seed=0):
    """Source-shaped frame with valid codes and a real risk signal."""
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    limit = rng.choice([20_000, 50_000, 100_000, 200_000, 500_000], size=n)
    p_late = 1 / (1 + np.exp(-(risk - 1.5)))
    cols = {"ID": np.arange(1, n + 1), "LIMIT_BAL": limit,
            "SEX": rng.integers(1, 3, n), "EDUCATION": rng.integers(1, 5, n),
            "MARRIAGE": rng.integers(1, 4, n), "AGE": rng.integers(21, 70, n)}
    for name in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        late = rng.random(n) < p_late
        cols[name] = np.where(late, rng.integers(1, 4, n), rng.choice([-2, -1, 0], n))
    util = np.clip(rng.beta(2, 3, n) + 0.2 * risk, 0, 1.2)
    for k in range(1, 7):
        cols[f"BILL_AMT{k}"] = np.round(limit * util * rng.uniform(0.8, 1.2, n))
    for k in range(1, 7):
        share = np.clip(rng.uniform(0, 0.5, n) - 0.1 * risk, 0, 1)
        cols[f"PAY_AMT{k}"] = np.round(cols[f"BILL_AMT{k}"].clip(0) * share)
    p_default = 1 / (1 + np.exp(-(-1.3 + 1.2 * risk)))
    cols["default payment next month"] = (rng.random(n) < p_default).astype(int)
    return pd.DataFrame(cols)[RAW_COLUMNS]


def build_warehouse(df):
    con = connect(":memory:")
    load_raw(con, df)
    build(con)
    return con


@pytest.fixture
def raw_df():
    return synthetic_raw()


@pytest.fixture
def warehouse(raw_df):
    con = build_warehouse(raw_df)
    yield con
    con.close()
