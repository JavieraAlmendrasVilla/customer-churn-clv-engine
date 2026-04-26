"""Model training entry point: XGBoost churn classifier + BG/NBD CLV model.

Usage:
    python src/train.py
"""
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config
from src.features import build_rfm_features, get_feature_matrix
from src.utils import get_logger, save_metrics

logger = get_logger(__name__)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split features and labels into train / validation / test sets.

    Ratios are 70 / 15 / 15 by default (see :mod:`config`).
    Stratification preserves the churn class balance in every split.

    Args:
        X: Feature DataFrame indexed by ``customer_unique_id``.
        y: Binary target Series.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        random_state: Seed for reproducibility.

    Returns:
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_state, stratify=y
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_frac), random_state=random_state, stratify=y_temp
    )
    logger.info(
        "Split — train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Pipeline:
    """Train an XGBoost churn classifier wrapped in a sklearn Pipeline.

    The pipeline applies :class:`~sklearn.preprocessing.StandardScaler` before
    fitting :class:`~xgboost.XGBClassifier`.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (used for early-stopping eval set).
        y_val: Validation labels.

    Returns:
        Fitted :class:`~sklearn.pipeline.Pipeline`.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Compensate for class imbalance (churned=1 dominates in Olist)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    logger.info("Class balance — pos: %d | neg: %d | scale_pos_weight: %.3f", pos, neg, spw)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=50,
    )

    pipeline = Pipeline([("scaler", scaler), ("clf", clf)])
    val_auc = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])
    logger.info("Validation ROC-AUC: %.4f", val_auc)
    return pipeline


def train_clv_model(
    rfm_df: pd.DataFrame,
) -> Tuple[BetaGeoFitter, GammaGammaFitter]:
    """Fit BG/NBD and Gamma-Gamma CLV models using the ``lifetimes`` library.

    The BG/NBD model captures purchase frequency; the Gamma-Gamma model
    estimates average order value for repeat buyers.

    Args:
        rfm_df: DataFrame with ``frequency``, ``recency_days``,
                ``monetary_value``, and ``customer_age_days`` columns.

    Returns:
        Tuple of ``(BetaGeoFitter, GammaGammaFitter)``.
    """
    # lifetimes convention:
    #   frequency = repeat purchases (total distinct dates - 1, clipped at 0)
    #   recency   = customer age at last purchase (first → last, in weeks)
    #   T         = total customer age (first purchase → snapshot, in weeks)
    lt_frequency = (rfm_df["frequency"] - 1).clip(lower=0)
    lt_recency = rfm_df["customer_lifespan_days"] / 7.0
    lt_T = rfm_df["customer_age_days"] / 7.0

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(lt_frequency, lt_recency, lt_T)

    # Gamma-Gamma requires customers with at least one repeat purchase
    repeat = rfm_df[rfm_df["frequency"] > 1]
    lt_freq_repeat = (repeat["frequency"] - 1).clip(lower=0)
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(lt_freq_repeat, repeat["monetary_value"])

    logger.info("CLV models (BG/NBD + Gamma-Gamma) fitted on %d customers", len(rfm_df))
    return bgf, ggf


def save_clv_params(bgf: BetaGeoFitter, ggf: GammaGammaFitter) -> None:
    """Save BG/NBD and Gamma-Gamma fitted parameters as JSON files.

    Must be called before ``get_feature_matrix()`` because that function loads
    the BG/NBD params to compute P(alive) churn labels.

    Args:
        bgf: Fitted :class:`~lifetimes.BetaGeoFitter`.
        ggf: Fitted :class:`~lifetimes.GammaGammaFitter`.
    """
    import json

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bgf_path = config.CLV_BGNBD_MODEL_PATH.with_suffix(".json")
    ggf_path = config.CLV_GG_MODEL_PATH.with_suffix(".json")
    with open(bgf_path, "w") as f:
        json.dump(dict(bgf.params_), f)
    with open(ggf_path, "w") as f:
        json.dump(dict(ggf.params_), f)
    logger.info("CLV params saved → %s", config.MODELS_DIR)


def save_churn_model(pipeline: Pipeline) -> None:
    """Save the fitted churn Pipeline with joblib.

    Args:
        pipeline: Fitted sklearn :class:`~sklearn.pipeline.Pipeline`.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.CHURN_MODEL_PATH)
    logger.info("Churn model saved → %s", config.CHURN_MODEL_PATH)


if __name__ == "__main__":
    logger.info("=== Training pipeline start ===")

    # ── Step 1: fit CLV models and save params ────────────────────────────────
    # Must happen first — get_feature_matrix() uses BG/NBD P(alive) as the
    # churn label, so the params file must exist before labels are built.
    rfm_df = build_rfm_features()
    bgf, ggf = train_clv_model(rfm_df)
    save_clv_params(bgf, ggf)

    # ── Step 2: build BG/NBD-labelled feature matrix and train XGBoost ───────
    X, y = get_feature_matrix()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    churn_pipeline = train_churn_model(X_train, y_train, X_val, y_val)
    save_churn_model(churn_pipeline)

    # ── Step 3: evaluate and log ──────────────────────────────────────────────
    test_auc = roc_auc_score(y_test, churn_pipeline.predict_proba(X_test)[:, 1])
    run_path = save_metrics({"test_roc_auc": test_auc}, "churn_xgb", config.RUNS_DIR)
    logger.info("Test ROC-AUC: %.4f — metrics → %s", test_auc, run_path)
    logger.info("=== Training pipeline complete ===")
