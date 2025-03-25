SELECT 
    p.product_id,
    p.name,
    p.description,
    p.price,
    c.name as category
FROM products p
JOIN categories c ON p.category_id = c.category_id
WHERE p.description ILIKE '%premium%'
    OR p.description ILIKE '%quality%'
ORDER BY p.price DESC; 