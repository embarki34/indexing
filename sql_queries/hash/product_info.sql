SELECT 
    p.product_id,
    p.name,
    p.description,
    p.price,
    p.stock_quantity,
    c.name as category_name
FROM products p
JOIN categories c ON p.category_id = c.category_id
WHERE p.product_id = 100; 