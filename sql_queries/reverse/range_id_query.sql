-- Favored Index Type: Reverse
-- Reason: Good for range queries on sequential IDs
-- Index Used: CREATE INDEX reverse_orders_id ON Orders USING BTREE (order_id) REVERSE;

SELECT order_id, order_date, status
FROM Orders
WHERE order_id BETWEEN 5000 AND 5100;

-- Required Index:
-- CREATE INDEX reverse_orders_id ON Orders USING BTREE (order_id) REVERSE; 