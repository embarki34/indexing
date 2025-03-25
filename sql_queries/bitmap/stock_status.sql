SELECT 
    CASE 
        WHEN stock_quantity = 0 THEN 'Out of Stock'
        WHEN stock_quantity < 10 THEN 'Low Stock'
        WHEN stock_quantity < 50 THEN 'Medium Stock'
        ELSE 'Well Stocked'
    END as stock_status,
    COUNT(*) as product_count
FROM products
GROUP BY stock_status; 