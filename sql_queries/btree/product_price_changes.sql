SELECT 
    p1.product_id,
    p1.name,
    p1.price as current_price,
    COUNT(DISTINCT o.order_id) as total_orders,
    MIN(oi.price) as min_sold_price,
    MAX(oi.price) as max_sold_price
FROM products p1
LEFT JOIN orderitems oi ON p1.product_id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.order_id
GROUP BY p1.product_id, p1.name, p1.price
HAVING p1.price > AVG(oi.price)
ORDER BY p1.price DESC; 