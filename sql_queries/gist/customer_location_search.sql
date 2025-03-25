SELECT 
    u.user_id,
    u.first_name,
    u.last_name,
    u.shipping_address,
    COUNT(o.order_id) as total_orders,
    SUM(o.total_amount) as total_spent
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
WHERE u.shipping_address ILIKE '%California%'
GROUP BY u.user_id, u.first_name, u.last_name, u.shipping_address; 