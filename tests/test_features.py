"""Unit tests for src/features.py."""
import pandas as pd
import pytest

import config
from src.features import add_churn_label, FEATURE_COLS


class TestAddChurnLabel:
    """Tests for the add_churn_label function."""

    def test_churned_column_created(self, sample_rfm_df: pd.DataFrame) -> None:
        result = add_churn_label(sample_rfm_df)
        assert "churned" in result.columns

    def test_churned_values_binary(self, sample_rfm_df: pd.DataFrame) -> None:
        result = add_churn_label(sample_rfm_df)
        assert set(result["churned"].unique()).issubset({0, 1})

    def test_churn_based_on_recency(self, sample_rfm_df: pd.DataFrame) -> None:
        window = 180
        result = add_churn_label(sample_rfm_df, churn_window_days=window)
        # All rows with recency_days >= window must be labelled churned
        mask = sample_rfm_df["recency_days"] >= window
        assert (result.loc[mask, "churned"] == 1).all()
        assert (result.loc[~mask, "churned"] == 0).all()

    def test_custom_window(self, sample_rfm_df: pd.DataFrame) -> None:
        result_30 = add_churn_label(sample_rfm_df, churn_window_days=30)
        result_365 = add_churn_label(sample_rfm_df, churn_window_days=365)
        # A longer window should produce fewer or equal churned customers
        assert result_30["churned"].sum() >= result_365["churned"].sum()

    def test_original_df_not_mutated(self, sample_rfm_df: pd.DataFrame) -> None:
        original_cols = list(sample_rfm_df.columns)
        add_churn_label(sample_rfm_df)
        assert list(sample_rfm_df.columns) == original_cols


class TestFeatureCols:
    """Tests that FEATURE_COLS are present in the RFM DataFrame."""

    def test_feature_cols_present(self, sample_rfm_df: pd.DataFrame) -> None:
        for col in FEATURE_COLS:
            assert col in sample_rfm_df.columns, f"Missing feature column: {col}"

    def test_no_nulls_in_features(self, sample_rfm_df: pd.DataFrame) -> None:
        for col in FEATURE_COLS:
            assert sample_rfm_df[col].notna().all(), f"Nulls found in: {col}"
