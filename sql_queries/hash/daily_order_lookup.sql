SELECT 
    o.order_id,
    o.order_date,
    u.email,
    SUM(oi.quantity) as total_items,
    STRING_AGG(p.name, ', ') as products_ordered
FROM orders o
JOIN users u ON o.user_id = u.user_id
JOIN orderitems oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_id = 5000
GROUP BY o.order_id, o.order_date, u.email; 