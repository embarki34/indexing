-- Favored Index Type: B-Tree
-- Reason: Excellent for range queries and sorting operations
-- Index Used: CREATE INDEX btree_products_price ON Products USING BTREE (price);

SELECT product_id, name, price 
FROM Products 
WHERE price BETWEEN 50 AND 150
ORDER BY price; 

-- Required Index:
-- CREATE INDEX btree_products_price ON Products USING BTREE (price); 