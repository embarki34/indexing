SELECT 
    o.order_id,
    o.order_date,
    oi.product_id,
    p.name as product_name,
    oi.quantity,
    oi.price
FROM orders o
JOIN orderitems oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_id = 12345; 