SELECT 
    o.order_id,
    o.order_date,
    o.total_amount,
    u.email,
    COUNT(oi.order_item_id) as items_count
FROM orders o
JOIN users u ON o.user_id = u.user_id
JOIN orderitems oi ON o.order_id = oi.order_id
WHERE o.order_date BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY o.order_id, o.order_date, o.total_amount, u.email
ORDER BY o.order_date DESC; 