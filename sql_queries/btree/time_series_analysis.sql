-- Favored Index Type: B-Tree
-- Reason: Optimal for range queries on dates and sequential sorting operations

WITH MonthlyRevenue AS (
    SELECT 
        TO_CHAR(o.order_date, 'YYYY-MM') AS month,  -- Changed DATE_FORMAT to TO_CHAR for PostgreSQL
        c.category_id,
        c.name AS category_name,
        SUM(oi.quantity * oi.price) AS monthly_revenue,
        COUNT(DISTINCT o.order_id) AS order_count,
        COUNT(DISTINCT o.user_id) AS customer_count
    FROM 
        Orders o
    JOIN 
        OrderItems oi ON o.order_id = oi.order_id
    JOIN 
        Products p ON oi.product_id = p.product_id
    JOIN 
        Categories c ON p.category_id = c.category_id
    WHERE 
        o.status != 'Cancelled'
        AND o.order_date >= CURRENT_DATE - INTERVAL '24 months'  -- Adjusted for PostgreSQL
    GROUP BY 
        TO_CHAR(o.order_date, 'YYYY-MM'), c.category_id, c.name  -- Changed DATE_FORMAT to TO_CHAR
)
SELECT 
    mr.month,
    mr.category_name,
    mr.monthly_revenue,
    mr.order_count,
    mr.customer_count,
    LAG(mr.monthly_revenue, 1) OVER (PARTITION BY mr.category_id ORDER BY mr.month) AS prev_month_revenue,
    (mr.monthly_revenue - LAG(mr.monthly_revenue, 1) OVER (PARTITION BY mr.category_id ORDER BY mr.month)) / 
        NULLIF(LAG(mr.monthly_revenue, 1) OVER (PARTITION BY mr.category_id ORDER BY mr.month), 0) * 100 AS month_over_month_growth,
    (mr.monthly_revenue - LAG(mr.monthly_revenue, 12) OVER (PARTITION BY mr.category_id ORDER BY mr.month)) / 
        NULLIF(LAG(mr.monthly_revenue, 12) OVER (PARTITION BY mr.category_id ORDER BY mr.month), 0) * 100 AS year_over_year_growth,
    AVG(mr.monthly_revenue) OVER (PARTITION BY mr.category_id ORDER BY mr.month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS rolling_3_month_avg,
    SUM(mr.monthly_revenue) OVER (PARTITION BY mr.category_id ORDER BY mr.month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_revenue
FROM 
    MonthlyRevenue mr
ORDER BY 
    mr.category_id, mr.month;

-- Required Index:
-- CREATE INDEX btree_orders_date_status ON Orders (order_date, status);  -- PostgreSQL syntax
-- CREATE INDEX btree_products_category ON Products (category_id);  -- PostgreSQL syntax