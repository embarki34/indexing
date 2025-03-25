SELECT 
    user_id,
    email,
    first_name,
    last_name
FROM users
WHERE email LIKE '%@gmail.com'; 