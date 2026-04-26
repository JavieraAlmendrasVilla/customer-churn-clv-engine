"""Central configuration — all paths and ML constants live here.

Import this module everywhere instead of hardcoding paths.
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Directory layout ──────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
DBT_DIR: Path = ROOT_DIR / "dbt"
MODELS_DIR: Path = ROOT_DIR / "models"
RUNS_DIR: Path = ROOT_DIR / "runs"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
SRC_DIR: Path = ROOT_DIR / "src"

# ── DuckDB warehouse ──────────────────────────────────────────────────────────
DUCKDB_PATH: Path = PROCESSED_DATA_DIR / "warehouse.duckdb"

# ── Raw source files ──────────────────────────────────────────────────────────
RAW_FILES: dict[str, Path] = {
    "customers": RAW_DATA_DIR / "olist_customers_dataset.csv",
    "orders": RAW_DATA_DIR / "olist_orders_dataset.csv",
    "order_items": RAW_DATA_DIR / "olist_order_items_dataset.csv",
    "order_payments": RAW_DATA_DIR / "olist_order_payments_dataset.csv",
    "order_reviews": RAW_DATA_DIR / "olist_order_reviews_dataset.csv",
    "products": RAW_DATA_DIR / "olist_products_dataset.csv",
    "sellers": RAW_DATA_DIR / "olist_sellers_dataset.csv",
    "geolocation": RAW_DATA_DIR / "olist_geolocation_dataset.csv",
    "product_category_translation": RAW_DATA_DIR / "product_category_name_translation.csv",
}

# ── ML constants ──────────────────────────────────────────────────────────────
CHURN_WINDOW_DAYS: int = 365
# Temporal split cutoff for churn model training.
# RFM features are computed from orders BEFORE this date.
# Churn label is based on whether the customer purchased in the
# CHURN_WINDOW_DAYS after this date. Must leave ≥180 days before
# the dataset end (~2018-09) to have enough label signal.
CHURN_CUTOFF_DATE: str = "2017-06-01"
RANDOM_STATE: int = 42
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ── Saved model artefacts ─────────────────────────────────────────────────────
CHURN_MODEL_PATH: Path = MODELS_DIR / "churn_xgb.pkl"
CLV_BGNBD_MODEL_PATH: Path = MODELS_DIR / "clv_bgnbd.pkl"
CLV_GG_MODEL_PATH: Path = MODELS_DIR / "clv_gg.pkl"
PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.pkl"

# ── Processed outputs ─────────────────────────────────────────────────────────
PREDICTIONS_PATH: Path = PROCESSED_DATA_DIR / "predictions.csv"
