SELECT 
    p1.product_id,
    p1.name,
    p1.description
FROM products p1, products p2
WHERE p1.product_id != p2.product_id
    AND similarity(p1.name, p2.name) > 0.3
ORDER BY similarity(p1.name, p2.name) DESC; 