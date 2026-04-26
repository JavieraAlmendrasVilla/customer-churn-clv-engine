-- Staging: olist_order_reviews_dataset

with source as (
    select * from {{ source('raw', 'raw_order_reviews') }}
),

renamed as (
    select
        review_id,
        order_id,
        review_score,
        review_comment_title,
        review_comment_message,
        review_creation_date::timestamp   as review_created_at,
        review_answer_timestamp::timestamp as review_answered_at
    from source
)

select * from renamed
