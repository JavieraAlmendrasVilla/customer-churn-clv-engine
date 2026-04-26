"""Load raw Olist CSVs into DuckDB to seed the warehouse.

Run once before executing ``dbt run``:

    python src/setup_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb

import config
from src.utils import get_logger

logger = get_logger(__name__)


def create_warehouse(db_path: Path = config.DUCKDB_PATH) -> None:
    """Create (or replace) the DuckDB warehouse and load all raw Olist CSVs.

    Each CSV is loaded as a table named ``raw_<key>`` where ``<key>`` matches
    the :data:`config.RAW_FILES` dictionary keys.

    Args:
        db_path: Path where the ``.duckdb`` file will be created or overwritten.
    """
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Creating DuckDB warehouse at %s", db_path)

    conn = duckdb.connect(str(db_path))
    try:
        for table_name, csv_path in config.RAW_FILES.items():
            if not csv_path.exists():
                logger.warning("CSV not found, skipping: %s", csv_path)
                continue
            duckdb_table = f"raw_{table_name}"
            logger.info("Loading %-35s → %s", csv_path.name, duckdb_table)
            conn.execute(f"""
                CREATE OR REPLACE TABLE {duckdb_table} AS
                SELECT * FROM read_csv_auto('{csv_path.as_posix()}', header = true)
            """)
    finally:
        conn.close()

    logger.info("Warehouse setup complete — %d tables loaded.", len(config.RAW_FILES))


if __name__ == "__main__":
    create_warehouse()
