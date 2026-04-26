"""KPI card row rendered at the top of the Streamlit dashboard."""
import pandas as pd
import streamlit as st


def render_kpi_cards(df: pd.DataFrame) -> None:
    """Render the four top-level KPI metric cards.

    Displays: Total Customers · Avg Churn Probability · Revenue at Risk · High-Risk Count.

    Args:
        df: Filtered predictions DataFrame containing ``churn_proba``,
            ``churn_risk_tier``, and ``predicted_clv`` columns.
    """
    total_customers = len(df)
    avg_churn_proba = df["churn_proba"].mean()
    high_risk = df[df["churn_risk_tier"] == "High"]
    high_risk_count = len(high_risk)
    revenue_at_risk = high_risk["predicted_clv"].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="Total Customers",
        value=f"{total_customers:,}",
        help="Unique customers with at least one delivered order in the Olist dataset.",
    )
    col2.metric(
        label="Avg Churn Probability",
        value=f"{avg_churn_proba:.1%}",
        help=(
            "XGBoost predicted probability that a customer is 'churned' — "
            "defined as BG/NBD P(alive) < 50%. Low average reflects that most "
            "single-purchase customers are not yet flagged as permanently inactive."
        ),
    )
    col3.metric(
        label="Revenue at Risk (R$)",
        value=f"R$ {revenue_at_risk:,.0f}",
        help=(
            "Sum of predicted 12-month CLV for High-Risk customers. "
            "Represents future revenue that may be lost without retention action."
        ),
    )
    col4.metric(
        label="High-Risk Customers",
        value=f"{high_risk_count:,}",
        help=(
            "Customers where the model predicts churn probability > 66%. "
            "These are the primary retention intervention targets."
        ),
    )
