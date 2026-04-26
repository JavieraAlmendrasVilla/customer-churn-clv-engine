-- Intermediate: aggregate order-level metrics to the customer grain.
-- Ephemeral — consumed by mart models, not stored as a table.

with orders as (
    select * from {{ ref('stg_orders') }}
),

customers as (
    select * from {{ ref('stg_customers') }}
),

payments as (
    select
        order_id,
        sum(payment_value) as order_revenue
    from {{ ref('stg_order_payments') }}
    group by order_id
),

reviews as (
    select
        order_id,
        avg(review_score) as avg_review_score
    from {{ ref('stg_order_reviews') }}
    group by order_id
),

order_enriched as (
    select
        o.order_id,
        c.customer_unique_id,
        o.order_status,
        o.order_purchase_at,
        o.order_delivered_at,
        coalesce(p.order_revenue, 0)   as order_revenue,
        r.avg_review_score,
        datediff(
            'day',
            o.order_purchase_at,
            coalesce(o.order_delivered_at, o.order_estimated_delivery_at)
        ) as delivery_days
    from orders o
    join customers c using (customer_id)
    left join payments p using (order_id)
    left join reviews r using (order_id)
    where o.order_status = 'delivered'
),

customer_agg as (
    select
        customer_unique_id,
        count(distinct order_id)                          as total_orders,
        sum(order_revenue)                                as total_revenue,
        avg(order_revenue)                                as avg_order_value,
        avg(avg_review_score)                             as avg_review_score,
        avg(delivery_days)                                as avg_delivery_days,
        min(order_purchase_at)::date                      as first_purchase_date,
        max(order_purchase_at)::date                      as last_purchase_date,
        datediff('day', min(order_purchase_at), max(order_purchase_at)) as customer_lifespan_days
    from order_enriched
    group by customer_unique_id
)

select * from customer_agg
