-- Favored Index Type: Bitmap
-- Reason: Ideal for low-cardinality columns

SELECT order_id, user_id, order_date, status
FROM Orders
WHERE status = 'Delivered';

-- Required Index:
-- CREATE BITMAP INDEX bitmap_orders_status ON Orders(status); 