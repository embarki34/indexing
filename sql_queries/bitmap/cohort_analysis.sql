-- Favored Index Type: Bitmap
-- Reason: Excellent for grouping by cohorts which have relatively few distinct values

WITH FirstPurchases AS (
    SELECT 
        u.user_id,
        TO_CHAR(MIN(o.order_date), 'YYYY-MM') AS cohort_month  -- Changed DATE_FORMAT to TO_CHAR
    FROM
        Users u
    JOIN
        Orders o ON u.user_id = o.user_id
    GROUP BY
        u.user_id
),
CustomerOrders AS (
    SELECT
        fp.user_id,
        fp.cohort_month,
        TO_CHAR(o.order_date, 'YYYY-MM') AS order_month,  -- Changed DATE_FORMAT to TO_CHAR
        EXTRACT(YEAR FROM AGE(o.order_date, TO_DATE(fp.cohort_month, 'YYYY-MM'))) * 12 +  -- Calculate month difference
        EXTRACT(MONTH FROM AGE(o.order_date, TO_DATE(fp.cohort_month, 'YYYY-MM'))) AS month_number
    FROM
        FirstPurchases fp
    JOIN
        Orders o ON fp.user_id = o.user_id
),
CohortSize AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT user_id) AS num_users
    FROM
        FirstPurchases
    GROUP BY
        cohort_month
),
CohortRetention AS (
    SELECT
        co.cohort_month,
        co.month_number,
        COUNT(DISTINCT co.user_id) AS num_users
    FROM
        CustomerOrders co
    GROUP BY
        co.cohort_month, co.month_number
)
SELECT
    cr.cohort_month,
    cs.num_users AS cohort_size,
    cr.month_number,
    cr.num_users AS returning_users,
    (cr.num_users::float / cs.num_users) * 100 AS retention_rate  -- Ensure float division
FROM
    CohortRetention cr
JOIN
    CohortSize cs ON cr.cohort_month = cs.cohort_month
WHERE
    cr.month_number <= 12
ORDER BY
    cr.cohort_month, cr.month_number;

-- Required Index:
-- CREATE INDEX bitmap_orders_month ON Orders(TO_CHAR(order_date, 'YYYY-MM'));  -- Updated for PostgreSQLs