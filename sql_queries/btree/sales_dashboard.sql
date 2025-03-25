-- Favored Index Type: B-Tree
-- Reason: Excellent for date ranges, sorting, and multiple aggregations over time periods

WITH DailySales AS (
    SELECT
        DATE(o.order_date) AS sale_date,
        EXTRACT(YEAR FROM o.order_date) AS year,
        EXTRACT(MONTH FROM o.order_date) AS month,
        EXTRACT(DOW FROM o.order_date) + 1 AS day_of_week,  -- Adjusted for PostgreSQL
        EXTRACT(DOY FROM o.order_date) AS day_of_year,
        EXTRACT(QUARTER FROM o.order_date) AS quarter,
        COUNT(DISTINCT o.order_id) AS order_count,
        COUNT(DISTINCT o.user_id) AS customer_count,
        SUM(o.total_amount) AS revenue,
        AVG(o.total_amount) AS avg_order_value,
        SUM(oi.quantity) AS units_sold
    FROM
        Orders o
    JOIN
        OrderItems oi ON o.order_id = oi.order_id
    WHERE
        o.status != 'Cancelled'
        AND o.order_date >= CURRENT_DATE - INTERVAL '3 years'
    GROUP BY
        DATE(o.order_date),  -- Include the same expression used in SELECT
        EXTRACT(YEAR FROM o.order_date),  -- Added to GROUP BY
        EXTRACT(MONTH FROM o.order_date),  -- Added to GROUP BY
        EXTRACT(DOW FROM o.order_date),  -- Added to GROUP BY
        EXTRACT(DOY FROM o.order_date),  -- Added to GROUP BY
        EXTRACT(QUARTER FROM o.order_date)  -- Added to GROUP BY
)
SELECT
    ds.*,
    AVG(revenue) OVER (ORDER BY sale_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7day_avg_revenue,
    AVG(revenue) OVER (ORDER BY sale_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS rolling_30day_avg_revenue,
    LAG(revenue, 365) OVER (ORDER BY sale_date) AS revenue_last_year,
    CASE
        WHEN LAG(revenue, 365) OVER (ORDER BY sale_date) IS NOT NULL THEN
            (revenue - LAG(revenue, 365) OVER (ORDER BY sale_date)) /
            NULLIF(LAG(revenue, 365) OVER (ORDER BY sale_date), 0) * 100
        ELSE NULL
    END AS yoy_revenue_growth,
    AVG(revenue) OVER (PARTITION BY day_of_week) AS avg_revenue_by_day_of_week,
    AVG(revenue) OVER (PARTITION BY month) AS avg_revenue_by_month,
    AVG(revenue) OVER (PARTITION BY quarter) AS avg_revenue_by_quarter,
    revenue / NULLIF(AVG(revenue) OVER (PARTITION BY day_of_week), 0) AS day_of_week_index,
    revenue / NULLIF(AVG(revenue) OVER (PARTITION BY month), 0) AS month_index,
    DENSE_RANK() OVER (PARTITION BY year, month ORDER BY revenue DESC) AS day_rank_in_month,
    DENSE_RANK() OVER (PARTITION BY year ORDER BY revenue DESC) AS day_rank_in_year
FROM
    DailySales ds
ORDER BY
    sale_date DESC;

-- Required Index:
-- CREATE INDEX btree_orders_date_status ON Orders (order_date, status);  -- PostgreSQL syntax
-- CREATE INDEX btree_orderitems_order ON OrderItems (order_id);  -- PostgreSQL syntax