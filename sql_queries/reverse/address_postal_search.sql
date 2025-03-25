SELECT 
    SUBSTRING(shipping_address FROM '\d{5}$') as postal_code,
    COUNT(*) as customer_count,
    STRING_AGG(DISTINCT first_name || ' ' || last_name, ', ' LIMIT 5) as sample_customers
FROM users
WHERE shipping_address ~ '\d{5}$'
GROUP BY postal_code
ORDER BY customer_count DESC; 
    SUBSTRING(shipping_address FROM '\d{5}$') as postal_code,
    COUNT(*) as customer_count,
    STRING_AGG(DISTINCT first_name || ' ' || last_name, ', ' LIMIT 5) as sample_customers
FROM users
WHERE shipping_address ~ '\d{5}$'
GROUP BY postal_code
ORDER BY customer_count DESC;