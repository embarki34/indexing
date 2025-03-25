SELECT 
    u.user_id,
    DATE_TRUNC('month', o.order_date) as month,
    COUNT(o.order_id) as orders_per_month,
    AVG(o.total_amount) as avg_order_value,
    SUM(oi.quantity) as total_items
FROM users u
JOIN orders o ON u.user_id = o.user_id
JOIN orderitems oi ON o.order_id = oi.order_id
GROUP BY u.user_id, month
ORDER BY u.user_id, month; 