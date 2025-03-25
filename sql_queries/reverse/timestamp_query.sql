-- Favored Index Type: Reverse
-- Reason: Efficient for timestamp-based queries with frequent inserts

SELECT order_id, user_id, order_date, status
FROM Orders
WHERE order_date = '2024-03-15';

-- Required Index:
-- CREATE INDEX reverse_orders_date ON Orders USING BTREE (order_date) REVERSE; 