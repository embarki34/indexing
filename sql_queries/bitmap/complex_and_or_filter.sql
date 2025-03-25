-- Favored Index Type: Bitmap
-- Reason: Efficient for complex AND/OR operations

SELECT p.product_id, p.name, p.price, p.stock_quantity
FROM Products p
WHERE (p.category_id = 1 OR p.category_id = 2)
AND (p.stock_quantity < 10 OR p.stock_quantity > 100);

-- Required Index:
-- CREATE BITMAP INDEX bitmap_products_category ON Products(category_id);
-- CREATE BITMAP INDEX bitmap_products_stock_level ON Products(
--     CASE 
--         WHEN stock_quantity < 10 THEN 'Low'
--         WHEN stock_quantity BETWEEN 10 AND 50 THEN 'Medium'
--         WHEN stock_quantity > 50 THEN 'High'
--     END
-- ); 