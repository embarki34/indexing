-- Favored Index Type: Hash
-- Reason: Efficient for joins with exact matches

SELECT o.order_id, o.order_date, p.name as product_name
FROM Orders o
JOIN OrderItems oi ON o.order_id = oi.order_id
JOIN Products p ON oi.product_id = p.product_id
WHERE o.order_id = 12345;

-- Required Index:
-- CREATE INDEX hash_orders_id ON Orders USING HASH (order_id); 