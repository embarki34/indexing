SELECT 
    DATE_TRUNC('month', order_date) as month,
    status,
    COUNT(*) as orders
FROM orders
GROUP BY month, status
ORDER BY month, status; 