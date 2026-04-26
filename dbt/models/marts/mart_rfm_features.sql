-- Mart: RFM + behavioural feature table.
-- One row per customer_unique_id, consumed directly by src/features.py.

with customer_orders as (
    select * from {{ ref('int_customer_orders') }}
),

snapshot as (
    select max(last_purchase_date) as snapshot_date from customer_orders
),

rfm as (
    select
        co.customer_unique_id,
        s.snapshot_date,

        -- Recency: days since last purchase
        datediff('day', co.last_purchase_date, s.snapshot_date)   as recency_days,

        -- Frequency: number of distinct orders
        co.total_orders                                            as frequency,

        -- Monetary: total revenue
        co.total_revenue                                           as monetary_value,

        -- Derived features
        co.avg_order_value,
        co.avg_review_score,
        co.avg_delivery_days,
        co.customer_lifespan_days,
        datediff('day', co.first_purchase_date, s.snapshot_date)   as customer_age_days,
        co.first_purchase_date,
        co.last_purchase_date
    from customer_orders co, snapshot s
)

select * from rfm
