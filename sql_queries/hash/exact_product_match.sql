-- Favored Index Type: Hash
-- Reason: Efficient for exact primary key lookups

SELECT product_id, name, price
FROM Products
WHERE product_id = 5423;

-- Required Index:
-- CREATE INDEX hash_products_id ON Products USING HASH (product_id); 