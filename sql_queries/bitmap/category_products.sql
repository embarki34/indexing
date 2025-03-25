SELECT 
    c.name as category_name,
    COUNT(p.product_id) as product_count,
    AVG(p.price) as avg_price
FROM categories c
LEFT JOIN products p ON c.category_id = p.category_id
GROUP BY c.category_id, c.name; 