SELECT 
    p.product_id,
    p.name,
    p.price,
    p.stock_quantity,
    COUNT(oi.order_item_id) as times_ordered
FROM products p
LEFT JOIN orderitems oi ON p.product_id = oi.product_id
WHERE p.name LIKE '% 2023'
GROUP BY p.product_id, p.name, p.price, p.stock_quantity; 