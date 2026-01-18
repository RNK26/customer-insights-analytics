-- Monthly revenue, with a running total across the whole two years.

WITH monthly AS (
    SELECT
        DATE_TRUNC('month', InvoiceDate) AS month,
        SUM(Revenue)                     AS monthly_revenue,
        COUNT(DISTINCT Invoice)          AS orders
    FROM sales
    GROUP BY 1
)
SELECT
    month,
    monthly_revenue,
    orders,
    SUM(monthly_revenue) OVER (
        ORDER BY month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total_revenue
FROM monthly
ORDER BY month;
