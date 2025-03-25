SELECT 
    u.user_id,
    u.email,
    COUNT(o.order_id) as order_count,
    SUM(o.total_amount) as total_spent
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id, u.email
ORDER BY total_spent DESC NULLS LAST; 