SELECT 
    c.name as category_name,
    DATE_TRUNC('month', o.order_date) as month,
    COUNT(DISTINCT o.order_id) as total_orders,
    SUM(oi.quantity) as items_sold,
    SUM(oi.quantity * oi.price) as revenue
FROM categories c
JOIN products p ON c.category_id = p.category_id
JOIN orderitems oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
GROUP BY c.name, month
ORDER BY c.name, month; 