"""Model evaluation: classification metrics, ROC curve, and SHAP explainability.

Usage:
    python src/evaluate.py
"""
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

import config
from src.features import get_feature_matrix
from src.train import split_data
from src.utils import get_logger, save_metrics

logger = get_logger(__name__)


def compute_churn_metrics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Compute classification metrics for the churn model on the held-out test set.

    Args:
        pipeline: Fitted churn :class:`~sklearn.pipeline.Pipeline`.
        X_test: Test feature DataFrame.
        y_test: True binary test labels.

    Returns:
        Dictionary with keys ``roc_auc``, ``average_precision``, and
        ``classification_report`` (nested dict).
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    metrics: Dict[str, Any] = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    logger.info("Test ROC-AUC: %.4f | AP: %.4f", metrics["roc_auc"], metrics["average_precision"])
    return metrics


def get_roc_curve_data(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve arrays suitable for Plotly or matplotlib plotting.

    Args:
        pipeline: Fitted churn :class:`~sklearn.pipeline.Pipeline`.
        X_test: Test feature DataFrame.
        y_test: True binary test labels.

    Returns:
        Tuple of ``(fpr, tpr, auc_score)``.
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    return fpr, tpr, auc


def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> Tuple[shap.TreeExplainer, np.ndarray, pd.DataFrame]:
    """Compute SHAP values for the XGBoost classifier inside the pipeline.

    Scales features through the pipeline's scaler before computing SHAP
    so that feature names are preserved.

    Args:
        pipeline: Fitted sklearn Pipeline containing a ``scaler`` and ``clf`` step.
        X: Feature DataFrame to explain.
        max_samples: Maximum rows to explain (for speed).

    Returns:
        Tuple of ``(explainer, shap_values, X_sample_scaled)``.
    """
    clf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]

    X_scaled = pd.DataFrame(
        scaler.transform(X), columns=X.columns, index=X.index
    )
    X_sample = X_scaled.sample(
        min(max_samples, len(X_scaled)), random_state=config.RANDOM_STATE
    )
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values, X_sample


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save a SHAP beeswarm summary plot to disk.

    Args:
        shap_values: SHAP values array (n_samples × n_features).
        X_sample: Scaled feature DataFrame that was explained.
        output_dir: Directory to write the PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    out = output_dir / "shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot → %s", out)


def plot_shap_waterfall(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    customer_idx: int,
    output_dir: Path,
) -> None:
    """Save a SHAP waterfall plot for a single customer to disk.

    Args:
        explainer: Fitted :class:`shap.TreeExplainer`.
        shap_values: SHAP values array.
        X_sample: Scaled feature DataFrame that was explained.
        customer_idx: Row index within ``X_sample`` to plot.
        output_dir: Directory to write the PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[customer_idx],
            base_values=explainer.expected_value,
            data=X_sample.iloc[customer_idx].values,
            feature_names=list(X_sample.columns),
        ),
        show=False,
    )
    plt.tight_layout()
    out = output_dir / f"shap_waterfall_idx{customer_idx}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP waterfall plot → %s", out)


if __name__ == "__main__":
    X, y = get_feature_matrix()
    _, _, X_test, _, _, y_test = split_data(X, y)

    pipeline = joblib.load(config.CHURN_MODEL_PATH)

    metrics = compute_churn_metrics(pipeline, X_test, y_test)
    save_metrics(metrics, "churn_evaluation", config.RUNS_DIR)

    # Save ROC curve arrays for the Streamlit dashboard
    fpr, tpr, auc = get_roc_curve_data(pipeline, X_test, y_test)
    np.savez(config.RUNS_DIR / "roc_curve.npz", fpr=fpr, tpr=tpr, auc=np.array(auc))
    logger.info("ROC curve data → %s", config.RUNS_DIR / "roc_curve.npz")

    explainer, shap_values, X_sample = compute_shap_values(pipeline, X_test)
    plots_dir = config.RUNS_DIR / "plots"
    plot_shap_summary(shap_values, X_sample, plots_dir)
    plot_shap_waterfall(explainer, shap_values, X_sample, 0, plots_dir)

    logger.info("Evaluation complete.")
