SELECT 
    c.name as category,
    CASE 
        WHEN p.stock_quantity = 0 THEN 'Out of Stock'
        WHEN p.stock_quantity <= 10 THEN 'Critical'
        WHEN p.stock_quantity <= 50 THEN 'Low'
        WHEN p.stock_quantity <= 100 THEN 'Medium'
        ELSE 'High'
    END as stock_level,
    COUNT(*) as product_count,
    SUM(p.stock_quantity) as total_stock
FROM products p
JOIN categories c ON p.category_id = c.category_id
GROUP BY c.name, stock_level
ORDER BY c.name, stock_level; 