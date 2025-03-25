SELECT 
    SUBSTRING(email FROM '@(.*)$') as email_domain,
    COUNT(*) as user_count,
    COUNT(DISTINCT o.order_id) as total_orders,
    SUM(o.total_amount) as total_revenue
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY email_domain
ORDER BY user_count DESC; 