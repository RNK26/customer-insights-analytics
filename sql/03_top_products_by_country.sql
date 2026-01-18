-- Top 5 products in each of the 8 biggest markets.
--
-- A single global bestseller list is just the UK list, since the UK is 85% of
-- revenue. Partitioning by country gives each market its own leaderboard.

WITH country_totals AS (
    SELECT Country, SUM(Revenue) AS country_revenue
    FROM sales
    GROUP BY Country
    ORDER BY country_revenue DESC
    LIMIT 8
),
product_by_country AS (
    SELECT
        s.Country,
        s.Description        AS product,
        SUM(s.Revenue)       AS product_revenue,
        SUM(s.Quantity)      AS units_sold
    FROM sales s
    INNER JOIN country_totals c ON s.Country = c.Country
    WHERE s.Description IS NOT NULL
    GROUP BY s.Country, s.Description
),
ranked AS (
    SELECT
        Country,
        product,
        product_revenue,
        units_sold,
        RANK() OVER (
            PARTITION BY Country
            ORDER BY product_revenue DESC
        ) AS rank_in_country,
        ROUND(100.0 * product_revenue
              / SUM(product_revenue) OVER (PARTITION BY Country), 2) AS pct_of_country_revenue
    FROM product_by_country
)
SELECT *
FROM ranked
WHERE rank_in_country <= 5   -- needs its own CTE; you can't filter on a
                             -- window function in the same query that defines it
ORDER BY Country, rank_in_country;
