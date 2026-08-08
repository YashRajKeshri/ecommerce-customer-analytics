-- ====================================================================
-- 04_CHURN_COHORTS.SQL
-- £780K+ Churn-Risk Cohort Identification & Inactivity Thresholds
-- Isolates lapsed high-value customers and calculates financial exposure
-- ====================================================================

WITH customer_spend_summary AS (
    SELECT
        t.CustomerID AS customer_id,
        t.Country AS country,
        COUNT(DISTINCT t.InvoiceNo) AS total_orders,
        ROUND(SUM(t.TotalAmount), 2) AS total_spend,
        MAX(t.InvoiceDate) AS last_order_date,
        MIN(t.InvoiceDate) AS first_order_date,
        DATE_DIFF(DATE(MAX(t.InvoiceDate)), DATE(MIN(t.InvoiceDate)), DAY) AS customer_tenure_days
    FROM `ecommerce_db.online_retail_cleaned` t
    GROUP BY t.CustomerID, t.Country
),

customer_inactivity AS (
    SELECT
        c.customer_id,
        c.country,
        c.total_orders,
        c.total_spend,
        c.customer_tenure_days,
        DATE_DIFF(
            DATE((SELECT MAX(InvoiceDate) FROM `ecommerce_db.online_retail_cleaned`)),
            DATE(c.last_order_date),
            DAY
        ) AS days_since_last_order,
        ROUND(c.total_spend / c.total_orders, 2) AS avg_order_value
    FROM customer_spend_summary c
),

-- Flag £780K Churn Exposure Cohort: Customers inactive > 60 days with notable past spend
at_risk_cohort AS (
    SELECT
        customer_id,
        country,
        total_orders,
        total_spend AS revenue_at_risk,
        days_since_last_order,
        avg_order_value,
        CASE
            WHEN days_since_last_order >= 90 AND total_spend >= 1000 THEN 'Critical High-Value Churn'
            WHEN days_since_last_order >= 60 AND total_spend >= 500 THEN 'High Risk Lapsed'
            WHEN days_since_last_order >= 60 THEN 'Moderate Risk Lapsed'
            ELSE 'Healthy / Active'
        END AS risk_tier
    FROM customer_inactivity
    WHERE days_since_last_order >= 60
)

-- Summary of the £780k+ Financial Exposure
SELECT
    risk_tier,
    COUNT(DISTINCT customer_id) AS at_risk_accounts,
    ROUND(SUM(revenue_at_risk), 2) AS total_revenue_at_risk,
    ROUND(AVG(revenue_at_risk), 2) AS avg_historical_spend,
    ROUND(AVG(days_since_last_order), 1) AS avg_days_inactive
FROM at_risk_cohort
GROUP BY risk_tier
ORDER BY total_revenue_at_risk DESC;
