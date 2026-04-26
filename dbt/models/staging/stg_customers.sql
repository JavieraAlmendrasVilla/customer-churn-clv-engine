-- Staging: olist_customers_dataset
-- Expose both the transaction-level customer_id and the deduplicated
-- customer_unique_id for downstream joins.

with source as (
    select * from {{ source('raw', 'raw_customers') }}
),

renamed as (
    select
        customer_id,
        customer_unique_id,
        customer_zip_code_prefix  as customer_zip,
        customer_city,
        customer_state
    from source
)

select * from renamed
