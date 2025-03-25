-- Favored Index Type: Hash
-- Reason: Efficient for exact user_id lookups

SELECT 
    u.user_id,
    u.first_name,
    u.last_name,
    u.email,
    o.order_id,
    o.order_date,
    o.total_amount,
    o.status
FROM 
    Users u
JOIN 
    Orders o ON u.user_id = o.user_id
WHERE 
    u.user_id = 5001
ORDER BY 
    o.order_date DESC;

-- Required Index:
-- CREATE INDEX hash_users_id ON Users USING HASH (user_id);
-- CREATE INDEX hash_orders_user ON Orders USING HASH (user_id); 