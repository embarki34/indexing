-- Favored Index Type: Hash
-- Reason: Best for exact match lookups of product details

SELECT 
    p.product_id,
    p.name,
    p.description,
    p.price,
    p.stock_quantity,
    c.name AS category_name,
    COUNT(DISTINCT oi.order_id) AS times_ordered,
    SUM(oi.quantity) AS total_quantity_sold
FROM 
    Products p
JOIN 
    Categories c ON p.category_id = c.category_id
LEFT JOIN 
    OrderItems oi ON p.product_id = oi.product_id
WHERE 
    p.product_id = 1502
GROUP BY 
    p.product_id, p.name, p.description, p.price, p.stock_quantity, c.name;

-- Required Index:
-- CREATE INDEX hash_products_id ON Products USING HASH (product_id);
-- CREATE INDEX hash_orderitems_product ON OrderItems USING HASH (product_id); 