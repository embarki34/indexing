-- Favored Index Type: Bitmap
-- Reason: Effective for queries with multiple conditions on low-cardinality columns

SELECT 
    p.product_id,
    p.name,
    c.name AS category_name,
    SUM(oi.quantity) AS total_units_sold,
    SUM(oi.quantity * oi.price) AS total_revenue,
    AVG(oi.price) AS avg_selling_price,
    COUNT(DISTINCT o.order_id) AS appearance_in_orders,
    COUNT(DISTINCT o.user_id) AS unique_customers,
    SUM(oi.quantity) / COUNT(DISTINCT o.order_id) AS avg_quantity_per_order,
    SUM(CASE WHEN o.status = 'Cancelled' THEN oi.quantity ELSE 0 END) AS cancelled_units,
    (SUM(CASE WHEN o.status = 'Cancelled' THEN oi.quantity ELSE 0 END) / NULLIF(SUM(oi.quantity), 0)) * 100 AS cancellation_rate,
    RANK() OVER (PARTITION BY c.category_id ORDER BY SUM(oi.quantity) DESC) AS rank_in_category,
    PERCENT_RANK() OVER (ORDER BY SUM(oi.quantity * oi.price) DESC) AS percentile_rank_overall
FROM 
    Products p
JOIN 
    Categories c ON p.category_id = c.category_id
LEFT JOIN 
    OrderItems oi ON p.product_id = oi.product_id
LEFT JOIN 
    Orders o ON oi.order_id = o.order_id
GROUP BY 
    p.product_id, p.name, c.name, c.category_id
ORDER BY 
    total_revenue DESC;

-- Required Index:
-- CREATE BITMAP INDEX bitmap_orders_status ON Orders(status);
-- CREATE BITMAP INDEX bitmap_products_category ON Products(category_id); 