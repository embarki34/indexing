-- Favored Index Type: B-Tree
-- Reason: Excellent for range queries with sorting
-- Index Used: CREATE INDEX btree_orders_date ON Orders USING BTREE (order_date);

SELECT order_id, user_id, order_date, total_amount
FROM Orders
WHERE order_date BETWEEN '2024-01-01' AND '2024-03-31'
ORDER BY order_date DESC;

-- Required Index:
-- CREATE INDEX btree_orders_date ON Orders USING BTREE (order_date); 