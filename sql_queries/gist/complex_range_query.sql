-- Favored Index Type: GiST
-- Reason: Good for complex data types and range operations

SELECT order_id, user_id, order_date, shipping_address
FROM Orders
WHERE shipping_address ILIKE '%usa%';  -- Changed to ILIKE for case-insensitive substring search

-- Required Index:
-- CREATE INDEX gist_orders_address ON Orders USING GIST (shipping_address);