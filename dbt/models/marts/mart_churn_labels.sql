-- Mart: RFM features + binary churn label for model training.
-- Churn definition: no purchase in the last 180 days (see CLAUDE.md).

with rfm as (
    select * from {{ ref('mart_rfm_features') }}
)

select
    customer_unique_id,
    snapshot_date,
    recency_days,
    frequency,
    monetary_value,
    avg_order_value,
    avg_review_score,
    avg_delivery_days,
    customer_age_days,
    customer_lifespan_days,
    first_purchase_date,
    last_purchase_date,

    -- Binary churn label
    case when recency_days >= 180 then 1 else 0 end as churned
from rfm
