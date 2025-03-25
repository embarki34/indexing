 -- Favored Index Type: Bitmap
-- Reason: Ideal for low-cardinality columns like order status with only a few possible values

SELECT 
    status,
    COUNT(*) AS order_count,
    SUM(total_amount) AS total_revenue,
    AVG(total_amount) AS average_order_value,
    MIN(order_date) AS earliest_order,
    MAX(order_date) AS latest_order
FROM 
    Orders
WHERE 
    status IN ('Pending', 'Shipped', 'Delivered')
    AND order_date BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY 
    status
ORDER BY 
    order_count DESC;

-- Required Index:
-- CREATE BITMAP INDEX bitmap_orders_status ON Orders(status);