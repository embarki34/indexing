-- Favored Index Type: B-Tree
-- Reason: Efficient for sorting and date range filtering with multiple aggregations

SELECT 
    p.product_id,
    p.name,
    c.name AS category_name,
    SUM(CASE WHEN EXTRACT(MONTH FROM o.order_date) = 1 THEN oi.quantity ELSE 0 END) AS Jan_Sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM o.order_date) = 2 THEN oi.quantity ELSE 0 END) AS Feb_Sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM o.order_date) = 3 THEN oi.quantity ELSE 0 END) AS Mar_Sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM o.order_date) = 4 THEN oi.quantity ELSE 0 END) AS Apr_Sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM o.order_date) = 5 THEN oi.quantity ELSE 0 END) AS May_Sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM o.order_date) = 6 THEN oi.quantity ELSE 0 END) AS Jun_Sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM o.order_date) = 7 THEN oi.quantity ELSE 0 END) AS Jul_Sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM o.order_date) = 8 THEN oi.quantity ELSE 0 END) AS Aug_Sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM o.order_date) = 9 THEN oi.quantity ELSE 0 END) AS Sep_Sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM o.order_date) = 10 THEN oi.quantity ELSE 0 END) AS Oct_Sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM o.order_date) = 11 THEN oi.quantity ELSE 0 END) AS Nov_Sales,
    SUM(CASE WHEN EXTRACT(MONTH FROM o.order_date) = 12 THEN oi.quantity ELSE 0 END) AS Dec_Sales,
    SUM(oi.quantity) AS Total_Sales,
    SUM(oi.quantity * oi.price) AS Total_Revenue
FROM 
    Products p
JOIN 
    Categories c ON p.category_id = c.category_id
LEFT JOIN 
    OrderItems oi ON p.product_id = oi.product_id
LEFT JOIN 
    Orders o ON oi.order_id = o.order_id AND EXTRACT(YEAR FROM o.order_date) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY 
    p.product_id, p.name, c.name
ORDER BY 
    Total_Revenue DESC;

-- Required Index:
-- CREATE INDEX btree_orderitems_product ON OrderItems (product_id);  -- PostgreSQL syntax
-- CREATE INDEX btree_orders_date ON Orders (order_date);  -- PostgreSQL syntax