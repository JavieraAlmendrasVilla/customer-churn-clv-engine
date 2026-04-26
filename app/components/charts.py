"""Reusable Plotly chart components for the Streamlit dashboard.

All functions render directly into the Streamlit layout via ``st.plotly_chart``.
Do NOT use matplotlib here — Plotly only.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config

# Feature column names in training order — must match src/features.py FEATURE_COLS
_FEATURE_COLS = [
    "monetary_value",
    "avg_order_value",
    "avg_review_score",
    "avg_delivery_days",
    "n_distinct_categories",
    "avg_freight_value",
    "used_installments",
]

_FEATURE_LABELS = {
    "monetary_value": "Total Spend (R$)",
    "avg_order_value": "Avg Order Value (R$)",
    "avg_review_score": "Avg Review Score",
    "avg_delivery_days": "Avg Delivery Days",
    "n_distinct_categories": "# Distinct Categories",
    "avg_freight_value": "Avg Freight Value (R$)",
    "used_installments": "Used Installments",
}

_TIER_COLORS = {"Low": "#00CC96", "Medium": "#FFA15A", "High": "#EF553B"}


def plot_churn_distribution(df: pd.DataFrame) -> None:
    """Render churn probability histogram with risk tier colour bands.

    Args:
        df: Predictions DataFrame with ``churn_proba`` and ``churn_risk_tier``.
    """
    fig = px.histogram(
        df,
        x="churn_proba",
        color="churn_risk_tier",
        nbins=60,
        title="Churn Probability Distribution by Risk Tier",
        labels={"churn_proba": "Churn Probability", "churn_risk_tier": "Risk Tier"},
        color_discrete_map=_TIER_COLORS,
        category_orders={"churn_risk_tier": ["Low", "Medium", "High"]},
        opacity=0.85,
    )
    fig.update_layout(
        bargap=0.02,
        xaxis_tickformat=".0%",
        legend_title="Risk Tier",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_risk_tier_breakdown(df: pd.DataFrame) -> None:
    """Render a donut chart of customer counts per risk tier.

    Args:
        df: Predictions DataFrame with ``churn_risk_tier`` column.
    """
    counts = (
        df["churn_risk_tier"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .reset_index()
    )
    counts.columns = ["Risk Tier", "Customers"]

    fig = px.pie(
        counts,
        names="Risk Tier",
        values="Customers",
        hole=0.5,
        title="Customers by Risk Tier",
        color="Risk Tier",
        color_discrete_map=_TIER_COLORS,
    )
    fig.update_traces(textinfo="percent+label+value", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def plot_clv_vs_churn(df: pd.DataFrame) -> None:
    """Render a CLV vs Churn Probability quadrant scatter chart.

    Filters to customers with CLV > 0 (repeat purchasers) so the chart
    is not dominated by the ~98% single-purchase zero-CLV cluster.

    Args:
        df: Predictions DataFrame with ``churn_proba``, ``predicted_clv``,
            and ``churn_risk_tier`` columns.
    """
    repeat = df[df["predicted_clv"] > 0].copy()

    if repeat.empty:
        st.info("No customers with CLV > 0 in the current filter selection.")
        return

    fig = px.scatter(
        repeat,
        x="churn_proba",
        y="predicted_clv",
        color="churn_risk_tier",
        color_discrete_map=_TIER_COLORS,
        title=f"CLV vs Churn Probability — Repeat Buyers ({len(repeat):,} customers)",
        labels={
            "churn_proba": "Churn Probability",
            "predicted_clv": "Predicted 12-Month CLV (R$)",
            "churn_risk_tier": "Risk Tier",
        },
        opacity=0.65,
        hover_data={"customer_unique_id": True},
    )

    # Quadrant lines
    fig.add_vline(x=0.5, line_dash="dash", line_color="grey", opacity=0.5)
    fig.add_hline(
        y=repeat["predicted_clv"].median(),
        line_dash="dash",
        line_color="grey",
        opacity=0.5,
    )

    median_clv = repeat["predicted_clv"].median()
    fig.add_annotation(x=0.15, y=repeat["predicted_clv"].max() * 0.95,
                       text="Safe — High Value", showarrow=False, font={"color": "#00CC96"})
    fig.add_annotation(x=0.75, y=repeat["predicted_clv"].max() * 0.95,
                       text="Danger — High Value", showarrow=False, font={"color": "#EF553B"})
    fig.add_annotation(x=0.15, y=median_clv * 0.3,
                       text="Safe — Low Value", showarrow=False, font={"color": "grey"})
    fig.add_annotation(x=0.75, y=median_clv * 0.3,
                       text="Monitor — Low Value", showarrow=False, font={"color": "#FFA15A"})

    fig.update_layout(xaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


def plot_clv_distribution(df: pd.DataFrame) -> None:
    """Render CLV histogram (CLV > 0 only) and avg CLV by risk tier side by side.

    Args:
        df: Predictions DataFrame with ``predicted_clv`` and ``churn_risk_tier``.
    """
    col1, col2 = st.columns(2)

    with col1:
        repeat = df[df["predicted_clv"] > 0]
        fig = px.histogram(
            repeat,
            x="predicted_clv",
            nbins=50,
            title=f"CLV Distribution — Repeat Buyers ({len(repeat):,})",
            labels={"predicted_clv": "Predicted 12-Month CLV (R$)"},
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        segment_stats = (
            df[df["predicted_clv"] > 0]
            .groupby("churn_risk_tier")["predicted_clv"]
            .agg(["mean", "sum", "count"])
            .reindex(["Low", "Medium", "High"])
            .reset_index()
            .rename(columns={"mean": "Avg CLV", "sum": "Total CLV", "count": "Customers"})
        )
        fig2 = px.bar(
            segment_stats,
            x="churn_risk_tier",
            y="Avg CLV",
            color="churn_risk_tier",
            title="Avg CLV by Risk Tier (Repeat Buyers)",
            labels={"Avg CLV": "Avg CLV (R$)", "churn_risk_tier": "Risk Tier"},
            color_discrete_map=_TIER_COLORS,
            text_auto=".2f",
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)


def plot_roc_curve() -> None:
    """Render the ROC curve from the most recent evaluation run.

    Loads pre-computed FPR/TPR arrays from the runs/ directory.
    Displays a placeholder message if no evaluation data is found.
    """
    roc_path = config.RUNS_DIR / "roc_curve.npz"
    if not roc_path.exists():
        st.info("No ROC data found. Run `python src/evaluate.py` to generate it.")
        return

    data = np.load(roc_path)
    fpr, tpr, auc = data["fpr"], data["tpr"], float(data["auc"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"XGBoost (AUC = {auc:.3f})",
                             line={"color": "#636EFA", "width": 2}))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                   line={"dash": "dash", "color": "grey"}, name="Random Baseline")
    )
    fig.update_layout(
        title="ROC Curve — Churn Classifier",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend={"x": 0.6, "y": 0.1},
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance() -> None:
    """Render a bar chart of XGBoost feature importances.

    Loads the fitted churn pipeline from models/. Feature names are mapped
    from the fixed FEATURE_COLS order used at training time.
    """
    if not config.CHURN_MODEL_PATH.exists():
        st.info("No model found. Run `python src/train.py` to train the model.")
        return

    pipeline = joblib.load(config.CHURN_MODEL_PATH)
    clf = pipeline.named_steps["clf"]
    importances = clf.feature_importances_

    labels = [_FEATURE_LABELS.get(f, f) for f in _FEATURE_COLS]

    fig = px.bar(
        x=importances,
        y=labels,
        orientation="h",
        title="Feature Importance (XGBoost Gain)",
        labels={"x": "Importance Score", "y": "Feature"},
        color=importances,
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
