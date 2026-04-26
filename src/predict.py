"""Batch prediction pipeline: churn probability scores + CLV estimates.

Usage:
    python src/predict.py
"""
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from sklearn.pipeline import Pipeline

import config
from src.features import build_behavioral_features, build_rfm_features
from src.utils import get_logger

logger = get_logger(__name__)

_RISK_BINS = [0.0, 0.33, 0.66, 1.0]
_RISK_LABELS = ["Low", "Medium", "High"]


def _require_model(path: Path) -> None:
    """Raise FileNotFoundError if a model artefact is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python src/train.py` first."
        )


def load_churn_model() -> Pipeline:
    """Load the serialised churn Pipeline from disk.

    Returns:
        Fitted :class:`~sklearn.pipeline.Pipeline`.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    _require_model(config.CHURN_MODEL_PATH)
    return joblib.load(config.CHURN_MODEL_PATH)


def load_clv_models() -> Tuple[BetaGeoFitter, GammaGammaFitter]:
    """Load CLV models from their JSON parameter files and reconstruct them.

    The lifetimes models are saved as ``*.json`` param dicts (not pickled)
    because their internal lambda closures are not serialisable by joblib.

    Returns:
        Tuple of ``(BetaGeoFitter, GammaGammaFitter)``.

    Raises:
        FileNotFoundError: If any model parameter file does not exist.
    """
    import json

    bgf_path = config.CLV_BGNBD_MODEL_PATH.with_suffix(".json")
    ggf_path = config.CLV_GG_MODEL_PATH.with_suffix(".json")
    _require_model(bgf_path)
    _require_model(ggf_path)

    bgf = BetaGeoFitter()
    bgf.params_ = json.loads(bgf_path.read_text())
    # lifetimes sets `predict` as an instance attribute inside fit(); restore it
    bgf.predict = bgf.conditional_expected_number_of_purchases_up_to_time

    ggf = GammaGammaFitter()
    ggf.params_ = json.loads(ggf_path.read_text())
    ggf.predict = ggf.conditional_expected_average_profit

    return bgf, ggf


def predict_churn(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Generate churn probability scores and risk tiers for all customers.

    Args:
        rfm_df: DataFrame with RFM feature columns.

    Returns:
        DataFrame with columns ``customer_unique_id``, ``churn_proba``,
        and ``churn_risk_tier`` (Low / Medium / High).
    """
    from src.features import FEATURE_COLS  # local to avoid circular at module level

    pipeline = load_churn_model()
    proba = pipeline.predict_proba(rfm_df[FEATURE_COLS])[:, 1]

    result = rfm_df[["customer_unique_id"]].copy()
    result["churn_proba"] = proba
    result["churn_risk_tier"] = pd.cut(
        proba, bins=_RISK_BINS, labels=_RISK_LABELS, include_lowest=True
    ).astype(str)
    return result


def predict_clv(rfm_df: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Estimate Customer Lifetime Value over a forecast horizon.

    Uses the BG/NBD model to predict future purchases and the Gamma-Gamma
    model to estimate average order value.

    Args:
        rfm_df: DataFrame with ``frequency``, ``recency_days``,
                ``customer_age_days``, and ``monetary_value`` columns.
        months: Forecast horizon in months (default 12).

    Returns:
        DataFrame with ``customer_unique_id`` and ``predicted_clv`` columns.
    """
    bgf, ggf = load_clv_models()

    # lifetimes convention (same as train.py)
    lt_frequency = (rfm_df["frequency"] - 1).clip(lower=0)
    lt_recency = rfm_df["customer_lifespan_days"] / 7.0
    lt_T = rfm_df["customer_age_days"] / 7.0

    result = rfm_df[["customer_unique_id"]].copy()
    result["predicted_clv"] = ggf.customer_lifetime_value(
        bgf,
        lt_frequency,
        lt_recency,
        lt_T,
        rfm_df["monetary_value"],
        time=months,
        freq="W",
    ).clip(lower=0)
    return result


def run_batch_predictions(
    output_path: Optional[Path] = config.PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Run the full batch prediction pipeline for all customers.

    Loads RFM features, scores churn probability, estimates CLV, merges
    results, and optionally writes a CSV artefact.

    Args:
        output_path: If provided, predictions are saved as CSV to this path.

    Returns:
        Merged predictions DataFrame.
    """
    # Behavioral features needed by the churn model (review score, delivery, etc.)
    behavioral_df = build_behavioral_features()
    # Full RFM (with customer_lifespan_days) needed by the CLV model
    rfm_df = build_rfm_features()

    churn_preds = predict_churn(behavioral_df)
    clv_preds = predict_clv(rfm_df)

    predictions = churn_preds.merge(clv_preds, on="customer_unique_id")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        logger.info("Predictions saved → %s", output_path)

    logger.info("Batch predictions complete for %d customers", len(predictions))
    return predictions


if __name__ == "__main__":
    run_batch_predictions()
