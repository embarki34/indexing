-- Favored Index Type: Hash
-- Reason: Optimal for exact equality comparisons on order IDs

SELECT 
    o.order_id,
    o.user_id,
    o.order_date,
    o.total_amount,
    u.email,
    COUNT(oi.order_item_id) AS item_count,
    SUM(oi.quantity) AS total_quantity
FROM 
    Orders o
JOIN 
    Users u ON o.user_id = u.user_id
JOIN 
    OrderItems oi ON o.order_id = oi.order_id
WHERE 
    o.order_id = 12345
GROUP BY 
    o.order_id, o.user_id, o.order_date, o.total_amount, u.email;

-- Required Index:
-- CREATE INDEX hash_orders_id ON Orders USING HASH (order_id);
-- CREATE INDEX hash_orderitems_order ON OrderItems USING HASH (order_id); 