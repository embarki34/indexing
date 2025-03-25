-- Favored Index Type: B-Tree
-- Reason: Excellent for range queries, sorting, and handling date comparisons

SELECT
    u.user_id,
    u.email,
    COUNT(DISTINCT o.order_id) AS total_orders,
    SUM(o.total_amount) AS total_spent,
    AVG(o.total_amount) AS avg_order_value,
    EXTRACT(DAY FROM AGE(MAX(o.order_date), MIN(o.order_date))) / NULLIF(COUNT(o.order_id), 0) AS avg_days_between_orders,  
    CASE
        WHEN COUNT(o.order_id) > 10 AND SUM(o.total_amount) > 5000 THEN 'Premium'
        WHEN COUNT(o.order_id) > 5 AND SUM(o.total_amount) > 2000 THEN 'Gold'
        WHEN COUNT(o.order_id) > 2 AND SUM(o.total_amount) > 1000 THEN 'Silver'
        ELSE 'Bronze'
    END AS customer_segment,
    DENSE_RANK() OVER (ORDER BY SUM(o.total_amount) DESC) AS spending_rank,
    MAX(o.order_date) AS last_order_date,
    EXTRACT(DAY FROM AGE(CURRENT_DATE, MAX(o.order_date))) AS days_since_last_order  
FROM
    Users u
LEFT JOIN
    Orders o ON u.user_id = o.user_id
GROUP BY
    u.user_id, u.email
HAVING
    COUNT(o.order_id) > 0
ORDER BY
    total_spent DESC;

-- Required Index:
-- CREATE INDEX btree_orders_user_date ON Orders (user_id, order_date);  -- PostgreSQL syntax
-- CREATE INDEX btree_orders_total ON Orders (total_amount);  -- PostgreSQL syntax