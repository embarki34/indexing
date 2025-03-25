SELECT 
    p.product_id,
    p.name,
    SUM(oi.quantity) as total_sold,
    SUM(oi.quantity * oi.price) as total_revenue,
    RANK() OVER (ORDER BY SUM(oi.quantity * oi.price) DESC) as revenue_rank
FROM products p
JOIN orderitems oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.name
ORDER BY total_revenue DESC; 