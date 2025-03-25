SELECT 
    product_id,
    name,
    description,
    price
FROM products
WHERE name ILIKE '%wireless%'
   OR description ILIKE '%wireless%'; 