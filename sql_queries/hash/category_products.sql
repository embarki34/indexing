-- Favored Index Type: Hash
-- Reason: Optimal for exact category lookups

SELECT 
    p.product_id,
    p.name,
    p.price,
    p.stock_quantity,
    p.created_at
FROM 
    Products p
WHERE 
    p.category_id = 5
ORDER BY 
    p.price ASC;

-- Required Index:
-- CREATE INDEX hash_products_category ON Products USING HASH (category_id); 