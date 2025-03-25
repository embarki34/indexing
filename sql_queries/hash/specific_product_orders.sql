SELECT 
    p.product_id,
    p.name,
    o.order_date,
    u.email as customer_email,
    oi.quantity,
    oi.price as sold_price
FROM products p
JOIN orderitems oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
JOIN users u ON o.user_id = u.user_id
WHERE p.product_id = 123; 