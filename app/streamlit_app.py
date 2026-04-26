"""Main Streamlit dashboard for the Customer Churn & CLV Engine.

Launch with:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

import config
from app.components.charts import (
    plot_churn_distribution,
    plot_clv_distribution,
    plot_clv_vs_churn,
    plot_feature_importance,
    plot_risk_tier_breakdown,
    plot_roc_curve,
)
from app.components.kpi_cards import render_kpi_cards
from src.utils import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Churn & CLV Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=3600)
def load_predictions() -> pd.DataFrame:
    """Load pre-computed batch predictions from disk (cached for 1 hour).

    Returns:
        Predictions DataFrame with churn scores and CLV estimates.

    Raises:
        FileNotFoundError: If predictions CSV has not been generated yet.
    """
    if not config.PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions not found at {config.PREDICTIONS_PATH}. "
            "Run `python src/predict.py` first."
        )
    return pd.read_csv(config.PREDICTIONS_PATH)


def main() -> None:
    """Render the full Streamlit dashboard."""
    st.title("Customer Churn & CLV Dashboard")
    st.caption(
        "Olist Brazilian E-Commerce · XGBoost churn classifier · BG/NBD + Gamma-Gamma CLV"
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        df = load_predictions()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    # ── How this model works ──────────────────────────────────────────────────
    with st.expander("How this model works", expanded=False):
        st.markdown("""
**Churn label — BG/NBD P(alive)**

Olist has a ~97% single-purchase rate, so a traditional "no purchase in N days"
label is near-degenerate. Instead, we use the industry-standard approach for
non-contractual settings:

1. The **BG/NBD model** (BetaGeoFitter) learns each customer's purchase rate and
   dropout probability from their frequency, recency, and tenure.
2. It computes **P(alive)** — the probability the customer is still "active" and
   has not permanently stopped buying.
3. Customers with **P(alive) < 50% are labelled churned = 1**.

**XGBoost classifier**

An XGBoost model is trained on **behavioural features only** — review score,
delivery time, spend level, category breadth, and payment behaviour.
RFM inputs (recency, frequency, tenure) are intentionally excluded to prevent
the model from trivially replicating BG/NBD (which would give AUC ≈ 1.0 for the
wrong reasons). The classifier learns which *satisfaction and engagement signals*
predict inactivity.  Result: **ROC-AUC ≈ 0.98**.

**CLV — Gamma-Gamma model**

The **Gamma-Gamma model** (GammaGammaFitter) estimates average order value for
repeat buyers and multiplies by BG/NBD expected future purchases over 12 months.
Single-purchase customers (≈97%) receive CLV ≈ 0 because the model requires at
least one repeat transaction to calibrate spend variance.
        """)

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.header("Filters")

    risk_options = ["All", "High", "Medium", "Low"]
    selected_risk = st.sidebar.selectbox("Churn Risk Tier", risk_options)

    min_clv = float(df["predicted_clv"].min())
    max_clv = float(df["predicted_clv"].max())
    clv_range = st.sidebar.slider(
        "Predicted CLV Range (R$)",
        min_value=min_clv,
        max_value=max_clv,
        value=(min_clv, max_clv),
        format="R$ %.2f",
    )

    # Apply filters
    filtered = df.copy()
    if selected_risk != "All":
        filtered = filtered[filtered["churn_risk_tier"] == selected_risk]
    filtered = filtered[filtered["predicted_clv"].between(clv_range[0], clv_range[1])]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"**{len(filtered):,}** customers match filters")
    st.sidebar.markdown("""
**Risk tiers**
- 🔴 **High** — churn prob > 66%
- 🟠 **Medium** — 33% – 66%
- 🟢 **Low** — < 33%
    """)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    render_kpi_cards(filtered)
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Churn Risk", "CLV Analysis", "Model Performance"])

    # ── Tab 1: Churn Risk ─────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
**Churn probability** is the XGBoost model's output: the probability a customer
matches the BG/NBD "dead" pattern. Most customers score near zero — they are
recent or have stable purchase histories. The tail above 66% (High-Risk) are
customers whose behavioural signals (low review scores, many categories browsed,
high freight sensitivity) strongly resemble dormant buyers.
        """)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            plot_churn_distribution(filtered)
        with col_b:
            plot_risk_tier_breakdown(filtered)

        st.subheader("Customer Detail Table")
        st.caption(
            "Sorted by churn probability (highest first). "
            "Hover the column headers for definitions."
        )
        st.dataframe(
            filtered[["customer_unique_id", "churn_proba", "churn_risk_tier", "predicted_clv"]]
            .sort_values("churn_proba", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=420,
            column_config={
                "customer_unique_id": st.column_config.TextColumn("Customer ID"),
                "churn_proba": st.column_config.ProgressColumn(
                    "Churn Probability",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    help="XGBoost P(churned). Thresholds: Low <33%, Medium 33-66%, High >66%.",
                ),
                "churn_risk_tier": st.column_config.TextColumn(
                    "Risk Tier",
                    help="Derived from churn probability thresholds (0.33 / 0.66).",
                ),
                "predicted_clv": st.column_config.NumberColumn(
                    "Predicted CLV (R$)",
                    format="R$ %.2f",
                    help=(
                        "12-month CLV from BG/NBD × Gamma-Gamma. "
                        "Zero for single-purchase customers."
                    ),
                ),
            },
        )

    # ── Tab 2: CLV Analysis ───────────────────────────────────────────────────
    with tab2:
        st.markdown("""
**Why is CLV zero for most customers?**

The Gamma-Gamma model requires at least **one repeat purchase** to estimate
spend variance. In Olist, ~97% of customers bought only once, so their CLV is
effectively R$ 0 — the model cannot predict future spend without a baseline repeat.

Only the **2,015 repeat buyers** (~2.2%) have non-zero CLV estimates.
The charts below focus on this cohort. The **quadrant chart** (bottom) is the
key business view: top-right customers (high CLV, high churn risk) are the
highest-priority retention targets.
        """)

        plot_clv_distribution(filtered)

        st.subheader("CLV vs Churn Risk — Quadrant View")
        st.caption(
            "Dashed lines: vertical = 50% churn threshold, "
            "horizontal = median CLV of repeat buyers."
        )
        plot_clv_vs_churn(filtered)

    # ── Tab 3: Model Performance ──────────────────────────────────────────────
    with tab3:
        st.markdown("""
**ROC-AUC** measures the model's ability to rank churned customers above active ones.
An AUC of 1.0 would be perfect; 0.5 is random. Our XGBoost achieves ≈ **0.984**,
meaning it almost always ranks a truly churned customer above an active one.

**Feature importance** (XGBoost gain) shows which signals drive the predictions.
*Number of distinct product categories* dominates — customers who explore many
categories tend to remain active, while narrow or one-off buyers are more likely
to go dormant. Spend level and average order value follow as secondary signals.
        """)

        col1, col2 = st.columns(2)
        with col1:
            plot_roc_curve()
        with col2:
            plot_feature_importance()


if __name__ == "__main__":
    main()
