SELECT DISTINCT
    p1.name as product1,
    p2.name as product2,
    p1.price as price1,
    p2.price as price2,
    similarity(p1.name, p2.name) as name_similarity
FROM products p1
JOIN products p2 ON p1.category_id = p2.category_id
    AND p1.product_id < p2.product_id
WHERE similarity(p1.name, p2.name) > 0.4
ORDER BY name_similarity DESC; 