-- Favored Index Type: Reverse
-- Reason: Efficient for joins in high-concurrency environments
-- Index Used: CREATE INDEX reverse_orders_id ON Orders USING BTREE (order_id) REVERSE;

SELECT o.order_id, o.order_date, u.email
FROM Orders o
JOIN Users u ON o.user_id = u.user_id
WHERE o.order_id = 45678;

-- Required Index:
-- CREATE INDEX reverse_orders_id ON Orders USING BTREE (order_id) REVERSE; 