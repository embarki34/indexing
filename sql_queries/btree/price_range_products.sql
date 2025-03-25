SELECT 
    p.name,
    p.price,
    c.name as category
FROM products p
JOIN categories c ON p.category_id = c.category_id
WHERE p.price BETWEEN 100 AND 500
ORDER BY p.price; 