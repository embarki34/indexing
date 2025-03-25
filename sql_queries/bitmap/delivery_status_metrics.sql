SELECT 
    status,
    COUNT(*) as order_count,
    AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - order_date))/86400) as avg_days_in_status,
    MIN(total_amount) as min_order_value,
    MAX(total_amount) as max_order_value,
    AVG(total_amount) as avg_order_value
FROM orders
GROUP BY status; 