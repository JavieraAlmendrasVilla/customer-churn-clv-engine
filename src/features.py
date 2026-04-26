"""RFM and behavioural feature engineering from the DuckDB warehouse.

Two entry points:
- ``build_rfm_features``   — full history snapshot, used by predict.py and CLV training.
- ``build_churn_dataset``  — temporal split, used by train.py and evaluate.py to avoid
                             leakage between recency_days and the churn label.
"""
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import pandas as pd
from lifetimes import BetaGeoFitter

import config
from src.utils import get_duckdb_conn, get_logger

logger = get_logger(__name__)

# ── Full-history RFM (for CLV training and batch scoring) ─────────────────────
_RFM_QUERY = """
WITH delivered AS (
    SELECT
        c.customer_unique_id,
        o.order_purchase_timestamp::DATE  AS purchase_date,
        op.payment_value
    FROM raw_orders o
    JOIN raw_customers c        USING (customer_id)
    JOIN raw_order_payments op  USING (order_id)
    WHERE o.order_status = 'delivered'
),
snapshot AS (
    SELECT MAX(purchase_date) AS snapshot_date FROM delivered
),
rfm AS (
    SELECT
        d.customer_unique_id,
        s.snapshot_date,
        DATEDIFF('day', MAX(d.purchase_date), s.snapshot_date)   AS recency_days,
        COUNT(DISTINCT d.purchase_date)                          AS frequency,
        SUM(d.payment_value)                                     AS monetary_value,
        MIN(d.purchase_date)                                     AS first_purchase_date,
        MAX(d.purchase_date)                                     AS last_purchase_date,
        DATEDIFF('day', MIN(d.purchase_date), s.snapshot_date)   AS customer_age_days,
        DATEDIFF('day', MIN(d.purchase_date), MAX(d.purchase_date)) AS customer_lifespan_days
    FROM delivered d, snapshot s
    GROUP BY d.customer_unique_id, s.snapshot_date
)
SELECT * FROM rfm
"""

# Features used by XGBoost churn classifier.
# Intentionally excludes recency_days / frequency / customer_age_days
# because those are the BG/NBD inputs that define the P(alive) label —
# including them would let XGBoost trivially replicate BG/NBD (AUC → 1.0).
# XGBoost instead learns which *behavioural* signals (satisfaction, delivery
# quality, spend level) predict a customer being "alive" beyond pure RFM.
FEATURE_COLS: list[str] = [
    "monetary_value",       # total spend — spending level, independent of recency
    "avg_order_value",      # average basket size
    "avg_review_score",     # satisfaction proxy
    "avg_delivery_days",    # fulfilment quality
    "n_distinct_categories",  # breadth of interest
    "avg_freight_value",    # shipping sensitivity
    "used_installments",    # payment commitment signal
]


def build_rfm_features(db_path: Path = config.DUCKDB_PATH) -> pd.DataFrame:
    """Compute RFM features over the full transaction history.

    Uses the dataset's max purchase date as the snapshot. Suitable for
    CLV model training and batch scoring of current customers.
    Not suitable for churn model training — use ``build_churn_dataset`` instead.

    Args:
        db_path: Path to the DuckDB warehouse file.

    Returns:
        DataFrame with one row per ``customer_unique_id``.
    """
    with get_duckdb_conn(db_path, read_only=True) as conn:
        df = conn.execute(_RFM_QUERY).df()

    logger.info("Built RFM features for %d customers", len(df))
    return df


# ── Temporal churn dataset (leakage-free) ─────────────────────────────────────

def build_churn_dataset(
    cutoff_date: str = config.CHURN_CUTOFF_DATE,
    churn_window_days: int = config.CHURN_WINDOW_DAYS,
    db_path: Path = config.DUCKDB_PATH,
) -> pd.DataFrame:
    """Build a leakage-free churn dataset using a temporal train/label split.

    Features are computed from orders **before** ``cutoff_date``.
    The churn label is 1 if the customer made **no** purchase in the
    ``churn_window_days`` days immediately following ``cutoff_date``.

    This separates the observation window (features) from the outcome window
    (label), preventing ``recency_days`` from directly encoding the label.

    Args:
        cutoff_date: ISO date string (``YYYY-MM-DD``) separating features from label.
        churn_window_days: Days after cutoff used to define active vs churned.
        db_path: Path to the DuckDB warehouse file.

    Returns:
        DataFrame with RFM features and a ``churned`` column (0 = active, 1 = churned).
    """
    query = f"""
    WITH delivered AS (
        SELECT
            c.customer_unique_id,
            o.order_id,
            o.order_purchase_timestamp::DATE  AS purchase_date,
            op.payment_value,
            op.payment_installments,
            oi.freight_value,
            r.review_score,
            p.product_category_name AS product_category,
            DATEDIFF(
                'day',
                o.order_purchase_timestamp,
                COALESCE(o.order_delivered_customer_date, o.order_estimated_delivery_date)
            ) AS delivery_days
        FROM raw_orders o
        JOIN raw_customers c         USING (customer_id)
        JOIN raw_order_payments op   USING (order_id)
        JOIN raw_order_items oi      USING (order_id)
        LEFT JOIN raw_order_reviews r USING (order_id)
        LEFT JOIN raw_products p     USING (product_id)
        WHERE o.order_status = 'delivered'
    ),

    -- Observation window: orders BEFORE the cutoff
    obs AS (
        SELECT * FROM delivered
        WHERE purchase_date < '{cutoff_date}'::DATE
    ),

    -- Label window: customers who purchased WITHIN churn_window_days after cutoff
    active_after AS (
        SELECT DISTINCT customer_unique_id
        FROM delivered
        WHERE purchase_date >= '{cutoff_date}'::DATE
          AND purchase_date <  '{cutoff_date}'::DATE + INTERVAL {churn_window_days} DAYS
    ),

    -- Aggregated features per customer from observation window
    rfm AS (
        SELECT
            o.customer_unique_id,
            '{cutoff_date}'::DATE                                              AS cutoff_date,
            DATEDIFF('day', MAX(o.purchase_date), '{cutoff_date}'::DATE)       AS recency_days,
            COUNT(DISTINCT o.order_id)                                         AS frequency,
            SUM(o.payment_value)                                               AS monetary_value,
            MIN(o.purchase_date)                                               AS first_purchase_date,
            MAX(o.purchase_date)                                               AS last_purchase_date,
            DATEDIFF('day', MIN(o.purchase_date), '{cutoff_date}'::DATE)       AS customer_age_days,
            AVG(o.payment_value)                                               AS avg_order_value,
            AVG(o.review_score)                                                AS avg_review_score,
            AVG(CASE WHEN o.delivery_days >= 0 THEN o.delivery_days END)       AS avg_delivery_days,
            COUNT(DISTINCT o.product_category)                                 AS n_distinct_categories,
            AVG(o.freight_value)                                               AS avg_freight_value,
            MAX(CASE WHEN o.payment_installments > 1 THEN 1 ELSE 0 END)       AS used_installments
        FROM obs o
        GROUP BY o.customer_unique_id
    )

    SELECT
        r.*,
        CASE WHEN a.customer_unique_id IS NULL THEN 1 ELSE 0 END AS churned
    FROM rfm r
    LEFT JOIN active_after a USING (customer_unique_id)
    """

    with get_duckdb_conn(db_path, read_only=True) as conn:
        df = conn.execute(query).df()

    churn_rate = df["churned"].mean()
    logger.info(
        "Churn dataset built — cutoff: %s | customers: %d | churn rate: %.1f%%",
        cutoff_date, len(df), churn_rate * 100,
    )
    return df


_BEHAVIORAL_QUERY = """
WITH delivered AS (
    SELECT
        c.customer_unique_id,
        o.order_id,
        o.order_purchase_timestamp::DATE  AS purchase_date,
        op.payment_value,
        op.payment_installments,
        oi.freight_value,
        r.review_score,
        p.product_category_name          AS product_category,
        DATEDIFF(
            'day',
            o.order_purchase_timestamp,
            COALESCE(o.order_delivered_customer_date, o.order_estimated_delivery_date)
        ) AS delivery_days
    FROM raw_orders o
    JOIN raw_customers c          USING (customer_id)
    JOIN raw_order_payments op    USING (order_id)
    LEFT JOIN raw_order_items oi       USING (order_id)
    LEFT JOIN raw_order_reviews r USING (order_id)
    LEFT JOIN raw_products p      USING (product_id)
    WHERE o.order_status = 'delivered'
),
snapshot AS (SELECT MAX(purchase_date) AS snapshot_date FROM delivered),
agg AS (
    SELECT
        d.customer_unique_id,
        s.snapshot_date,
        DATEDIFF('day', MAX(d.purchase_date), s.snapshot_date)  AS recency_days,
        COUNT(DISTINCT d.order_id)                              AS frequency,
        SUM(d.payment_value)                                    AS monetary_value,
        AVG(d.payment_value)                                    AS avg_order_value,
        DATEDIFF('day', MIN(d.purchase_date), s.snapshot_date)      AS customer_age_days,
        DATEDIFF('day', MIN(d.purchase_date), MAX(d.purchase_date)) AS customer_lifespan_days,
        MIN(d.purchase_date)                                        AS first_purchase_date,
        MAX(d.purchase_date)                                        AS last_purchase_date,
        AVG(d.review_score)                                     AS avg_review_score,
        AVG(CASE WHEN d.delivery_days >= 0 THEN d.delivery_days END) AS avg_delivery_days,
        COUNT(DISTINCT d.product_category)                      AS n_distinct_categories,
        AVG(d.freight_value)                                    AS avg_freight_value,
        MAX(CASE WHEN d.payment_installments > 1 THEN 1 ELSE 0 END) AS used_installments
    FROM delivered d, snapshot s
    GROUP BY d.customer_unique_id, s.snapshot_date
)
SELECT * FROM agg
"""


def build_behavioral_features(db_path: Path = config.DUCKDB_PATH) -> pd.DataFrame:
    """Compute full behavioral feature set over the entire transaction history.

    Includes RFM + review scores, delivery time, category breadth, and
    payment behaviour. Used as inputs to the BG/NBD-labelled churn model.

    Args:
        db_path: Path to the DuckDB warehouse file.

    Returns:
        DataFrame with one row per ``customer_unique_id``.
    """
    with get_duckdb_conn(db_path, read_only=True) as conn:
        df = conn.execute(_BEHAVIORAL_QUERY).df()

    logger.info("Built behavioral features for %d customers", len(df))
    return df


def get_feature_matrix(
    db_path: Path = config.DUCKDB_PATH,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix X and target y using BG/NBD P(alive) as the churn label.

    Olist has a ~97% single-purchase rate, making temporal-window churn labels
    near-degenerate. BG/NBD's ``conditional_probability_alive`` is the
    industry-standard churn score for non-contractual, low-repurchase settings.
    XGBoost extends the BG/NBD signal with behavioral features (review score,
    delivery time, category breadth) that the statistical model cannot use.

    A customer is labelled **churned = 1** when their BG/NBD P(alive) < 0.5.

    Args:
        db_path: Path to the DuckDB warehouse file.

    Returns:
        Tuple of ``(X, y)`` indexed by ``customer_unique_id``.

    Raises:
        FileNotFoundError: If the BG/NBD params file has not been created yet.
                           Run ``python src/train.py`` once to bootstrap.
    """
    bgf_path = config.CLV_BGNBD_MODEL_PATH.with_suffix(".json")
    if not bgf_path.exists():
        raise FileNotFoundError(
            f"BG/NBD params not found at {bgf_path}. "
            "Run `python src/train.py --clv-only` first, or use "
            "`build_churn_dataset()` for the temporal-split fallback."
        )

    bgf = BetaGeoFitter()
    bgf.params_ = json.loads(bgf_path.read_text())

    df = build_behavioral_features(db_path)

    # lifetimes convention:
    #   frequency = repeat purchases (total - 1, clipped at 0)
    #   recency   = customer age at time of last purchase (first → last, in weeks)
    #   T         = total customer age (first purchase → snapshot, in weeks)
    lt_frequency = (df["frequency"] - 1).clip(lower=0)
    lt_recency = df["customer_lifespan_days"] / 7.0
    lt_T = df["customer_age_days"] / 7.0

    p_alive = bgf.conditional_probability_alive(lt_frequency, lt_recency, lt_T)
    df["churned"] = (p_alive < 0.5).astype(int)
    churn_rate = df["churned"].mean()
    logger.info(
        "BG/NBD churn label (P(alive)<0.5) — rate: %.1f%% | customers: %d",
        churn_rate * 100, len(df),
    )

    X = df.set_index("customer_unique_id")[FEATURE_COLS]
    y = df.set_index("customer_unique_id")["churned"]
    return X, y


if __name__ == "__main__":
    X, y = get_feature_matrix()
    print(X.describe())
    print(f"\nChurn rate: {y.mean():.1%}")
