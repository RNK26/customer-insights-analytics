-- Month-over-month next to year-over-year growth.
--
-- MoM on its own is misleading for a gift retailer -- the Q4 spike makes
-- November look like a great month every year. The 12-month lag is the column
-- that says whether the business actually grew.

WITH monthly AS (
    SELECT
        DATE_TRUNC('month', InvoiceDate) AS month,
        SUM(Revenue)                     AS monthly_revenue
    FROM sales
    GROUP BY 1
)
SELECT
    month,
    monthly_revenue,

    LAG(monthly_revenue, 1) OVER (ORDER BY month) AS prev_month_revenue,
    ROUND(
        100.0 * (monthly_revenue - LAG(monthly_revenue, 1) OVER (ORDER BY month))
              / LAG(monthly_revenue, 1) OVER (ORDER BY month)
    , 1) AS mom_growth_pct,

    LAG(monthly_revenue, 12) OVER (ORDER BY month) AS same_month_last_year,
    ROUND(
        100.0 * (monthly_revenue - LAG(monthly_revenue, 12) OVER (ORDER BY month))
              / LAG(monthly_revenue, 12) OVER (ORDER BY month)
    , 1) AS yoy_growth_pct

FROM monthly
ORDER BY month;
