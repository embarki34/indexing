SELECT 
    u.user_id,
    u.email,
    o.order_id,
    o.order_date,
    o.total_amount
FROM users u
JOIN orders o ON u.user_id = o.user_id
WHERE u.user_id = 789
ORDER BY o.order_date DESC
LIMIT 1; 