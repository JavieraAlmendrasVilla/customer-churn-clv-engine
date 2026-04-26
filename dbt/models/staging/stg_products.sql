-- Staging: olist_products_dataset + product_category_name_translation
-- Joins English category names in at this layer.

with products as (
    select * from {{ source('raw', 'raw_products') }}
),

translation as (
    select * from {{ source('raw', 'raw_product_category_translation') }}
),

joined as (
    select
        p.product_id,
        coalesce(t.product_category_name_english, p.product_category_name) as product_category,
        p.product_name_lenght        as product_name_length,
        p.product_description_lenght as product_description_length,
        p.product_photos_qty,
        p.product_weight_g,
        p.product_length_cm,
        p.product_height_cm,
        p.product_width_cm
    from products p
    left join translation t using (product_category_name)
)

select * from joined
