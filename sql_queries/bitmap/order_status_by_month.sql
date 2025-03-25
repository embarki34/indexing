SELECT 
    DATE_TRUNC('month', order_date) as month,
    status,
    COUNT(*) as order_count,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_order_value
FROM orders
WHERE status IN ('Pending', 'Shipped', 'Delivered')
GROUP BY month, status
ORDER BY month DESC, status; 