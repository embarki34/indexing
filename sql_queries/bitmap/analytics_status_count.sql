
SELECT COUNT(*) as order_count, status
FROM Orders
WHERE order_date BETWEEN '2017-01-01' AND '2025-06-30'
GROUP BY status;

