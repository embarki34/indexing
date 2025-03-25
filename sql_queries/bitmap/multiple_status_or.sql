-- Favored Index Type: Bitmap
-- Reason: Excellent for OR conditions on low-cardinality columns
-- Index Used: CREATE BITMAP INDEX bitmap_orders_status ON Orders(status);

SELECT order_id, user_id, order_date, status
FROM Orders
WHERE status = 'Pending' OR status = 'Shipped';

-- Required Index:
-- CREATE BITMAP INDEX bitmap_orders_status ON Orders(status); 