"""Shared pytest fixtures for the test suite."""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture()
def sample_rfm_df() -> pd.DataFrame:
    """Minimal RFM DataFrame that mirrors the shape produced by build_rfm_features."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "customer_unique_id": [f"cust_{i:04d}" for i in range(n)],
        "snapshot_date": pd.Timestamp("2018-09-03"),
        "recency_days": rng.integers(0, 500, n),
        "frequency": rng.integers(1, 10, n),
        "monetary_value": rng.uniform(20.0, 2000.0, n),
        "customer_age_days": rng.integers(1, 800, n),
        "avg_order_value": rng.uniform(20.0, 500.0, n),
        "avg_review_score": rng.uniform(1.0, 5.0, n),
        "avg_delivery_days": rng.integers(1, 30, n),
        "customer_lifespan_days": rng.integers(0, 700, n),
        "first_purchase_date": pd.Timestamp("2017-01-01"),
        "last_purchase_date": pd.Timestamp("2018-06-01"),
    })
