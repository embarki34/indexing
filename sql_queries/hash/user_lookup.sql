SELECT 
    u.user_id,
    u.email,
    u.first_name,
    u.last_name,
    u.phone_number
FROM users u
WHERE u.email = 'customer@example.com'; 