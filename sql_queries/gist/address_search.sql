SELECT 
    user_id,
    first_name,
    last_name,
    shipping_address
FROM users
WHERE shipping_address ILIKE '%New York%'; 