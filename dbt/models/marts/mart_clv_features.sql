-- Mart: features formatted for the lifetimes BG/NBD + Gamma-Gamma models.
--
-- BG/NBD expects:
--   frequency  = number of repeat purchases  (total_orders - 1, clipped at 0)
--   recency    = time between first and last purchase (in weeks)
--   T          = time between first purchase and observation date (in weeks)
--
-- Gamma-Gamma expects:
--   frequency  = same as above (repeat buyers only)
--   monetary_value = mean revenue per repeat transaction

with rfm as (
    select * from {{ ref('mart_rfm_features') }}
)

select
    customer_unique_id,
    snapshot_date,

    -- BG/NBD inputs (weeks)
    greatest(frequency - 1, 0)                                          as frequency,
    datediff('day', first_purchase_date, last_purchase_date) / 7.0      as recency_weeks,
    customer_age_days / 7.0                                              as T_weeks,

    -- Gamma-Gamma input: average revenue per repeat transaction
    case
        when frequency > 1 then monetary_value / frequency
        else monetary_value
    end                                                                  as monetary_value,

    -- Pass-through for reference
    total_revenue                                                        as total_revenue
from rfm
-- Exclude customers with zero age (same-day single purchase edge case)
where customer_age_days > 0
