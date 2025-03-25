SELECT 
    status,
    COUNT(*) as order_count,
    SUM(total_amount) as total_revenue
FROM orders
GROUP BY status
ORDER BY order_count DESC; 