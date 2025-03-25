-- Favored Index Type: Hash
-- Reason: Good for exact matches on low-cardinality columns
-- Index Used: CREATE INDEX hash_orders_status ON Orders USING HASH (status);

SELECT order_id, user_id, order_date, status
FROM Orders
WHERE status = 'Shipped';

-- Required Index:
-- CREATE INDEX hash_orders_status ON Orders USING HASH (status); 