-- Favored Index Type: GiST
-- Reason: Excellent for full-text search operations

SELECT product_id, name, description
FROM Products
WHERE to_tsvector('english', description) @@ to_tsquery('english', 'oil');

-- Required Index:
-- CREATE INDEX gist_products_description ON Products USING GIST (to_tsvector('english', description));