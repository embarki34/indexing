-- Favored Index Type: B-Tree
-- Reason: Efficient for composite indexes with range and equality conditions
-- Index Used: CREATE INDEX btree_product_cat_price ON Products USING BTREE (category_id, price);

SELECT p.product_id, p.name, p.price, c.name as category_name
FROM Products p
JOIN Categories c ON p.category_id = c.category_id
WHERE p.category_id = 3 
AND p.price BETWEEN 20 AND 100
ORDER BY p.price;

-- Required Index:
-- CREATE INDEX btree_product_cat_price ON Products USING BTREE (category_id, price); 