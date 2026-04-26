"""Shared helper utilities used across all src modules."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def get_logger(name: str) -> logging.Logger:
    """Create a module-level logger with consistent formatting.

    Args:
        name: Logger name — pass ``__name__`` from the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def get_duckdb_conn(db_path: Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection to the warehouse file.

    Args:
        db_path: Absolute path to the ``.duckdb`` file.
        read_only: Open in read-only mode (safe for concurrent reads).

    Returns:
        An open :class:`duckdb.DuckDBPyConnection`.

    Raises:
        FileNotFoundError: If ``read_only=True`` and ``db_path`` does not exist.
    """
    if read_only and not db_path.exists():
        raise FileNotFoundError(
            f"DuckDB warehouse not found at {db_path}. Run `python src/setup_db.py` first."
        )
    return duckdb.connect(str(db_path), read_only=read_only)


def save_metrics(metrics: dict[str, Any], run_name: str, runs_dir: Path) -> Path:
    """Persist a metrics dictionary to a timestamped JSON file.

    Args:
        metrics: Mapping of metric names to values.
        run_name: Human-readable identifier for this run (e.g. ``"churn_xgb_v1"``).
        runs_dir: Directory where run JSON files are written.

    Returns:
        Path to the saved JSON file.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = runs_dir / f"{run_name}_{timestamp}.json"
    payload: dict[str, Any] = {"run_name": run_name, "timestamp": timestamp, **metrics}
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return out_path


def load_table(table_name: str, db_path: Path) -> pd.DataFrame:
    """Load an entire DuckDB table or view into a pandas DataFrame.

    Args:
        table_name: Name of the table or view to query.
        db_path: Path to the DuckDB warehouse file.

    Returns:
        DataFrame containing all rows from the table.
    """
    with get_duckdb_conn(db_path, read_only=True) as conn:
        return conn.execute(f"SELECT * FROM {table_name}").df()  # noqa: S608
