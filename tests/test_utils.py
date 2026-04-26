"""Unit tests for src/utils.py."""
import json
import tempfile
from pathlib import Path

import pytest

from src.utils import save_metrics


class TestSaveMetrics:
    """Tests for the save_metrics helper."""

    def test_creates_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            out = save_metrics({"roc_auc": 0.85}, "test_run", runs_dir)
            assert out.exists()
            assert out.suffix == ".json"

    def test_json_contains_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            metrics = {"roc_auc": 0.85, "avg_precision": 0.72}
            out = save_metrics(metrics, "test_run", runs_dir)
            payload = json.loads(out.read_text())
            assert payload["roc_auc"] == 0.85
            assert payload["avg_precision"] == 0.72

    def test_run_name_in_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            out = save_metrics({}, "my_experiment", runs_dir)
            assert "my_experiment" in out.name

    def test_creates_runs_dir_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "nested" / "runs"
            assert not runs_dir.exists()
            save_metrics({"x": 1}, "run", runs_dir)
            assert runs_dir.exists()
