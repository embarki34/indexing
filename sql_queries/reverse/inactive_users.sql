-- Favored Index Type: Reverse
-- Reason: Good for high-concurrency environments with sequential IDs and time-based lookups

SELECT 
    u.user_id,
    u.first_name,
    u.last_name,
    u.email,
    u.created_at AS registration_date,
    MAX(o.order_date) AS last_order_date,
    DATEDIFF(CURRENT_DATE(), MAX(o.order_date)) AS days_since_last_order,
    COUNT(DISTINCT o.order_id) AS total_orders,
    SUM(o.total_amount) AS lifetime_value,
    SUM(o.total_amount) / COUNT(DISTINCT o.order_id) AS average_order_value
FROM 
    Users u
LEFT JOIN 
    Orders o ON u.user_id = o.user_id
GROUP BY
    u.user_id,
    u.first_name,
    u.last_name,
    u.email,
    u.created_at
HAVING
    MAX(o.order_date) IS NULL
    OR DATEDIFF(CURRENT_DATE(), MAX(o.order_date)) > 90
ORDER BY
    days_since_last_order DESC NULLS FIRST;

-- Required Index:
-- CREATE INDEX rev_users_id ON Users USING REVERSE (user_id);
-- CREATE INDEX rev_orders_date ON Orders USING REVERSE (order_date);