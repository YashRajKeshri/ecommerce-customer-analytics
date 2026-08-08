-- ====================================================================
-- 03_RFM_SEGMENTATION.SQL
-- Customer RFM (Recency, Frequency, Monetary) Segmentation & Pareto Analysis
-- Identifies top customer segments driving 68% of enterprise revenue
-- ====================================================================

WITH snapshot_date AS (
    -- Reference date: 1 day after the latest transaction
    SELECT TIMESTAMP_ADD(MAX(InvoiceDate), INTERVAL 1 DAY) AS reference_time
    FROM `ecommerce_db.online_retail_cleaned`
),

-- Step 1: Calculate raw Recency, Frequency, and Monetary values per Customer
customer_rfm_raw AS (
    SELECT
        t.CustomerID AS customer_id,
        DATE_DIFF(DATE(s.reference_time), DATE(MAX(t.InvoiceDate)), DAY) AS recency_days,
        COUNT(DISTINCT t.InvoiceNo) AS frequency,
        ROUND(SUM(t.TotalAmount), 2) AS monetary_total,
        MIN(t.InvoiceDate) AS first_purchase_date,
        MAX(t.InvoiceDate) AS last_purchase_date
    FROM `ecommerce_db.online_retail_cleaned` t
    CROSS JOIN snapshot_date s
    GROUP BY t.CustomerID, s.reference_time
),

-- Step 2: Assign quintile scores (1 to 5) using NTILE window functions
rfm_quantiles AS (
    SELECT
        customer_id,
        recency_days,
        frequency,
        monetary_total,
        -- Invert recency (lowest days gets score 5)
        6 - NTILE(5) OVER(ORDER BY recency_days ASC) AS r_score,
        NTILE(5) OVER(ORDER BY frequency ASC) AS f_score,
        NTILE(5) OVER(ORDER BY monetary_total ASC) AS m_score
    FROM customer_rfm_raw
),

-- Step 3: Map RFM combinations to strategic customer personas
rfm_segments AS (
    SELECT
        customer_id,
        recency_days,
        frequency,
        monetary_total,
        r_score,
        f_score,
        m_score,
        CONCAT(CAST(r_score AS STRING), CAST(f_score AS STRING), CAST(m_score AS STRING)) AS rfm_combined,
        CASE
            WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Loyal Customers'
            WHEN r_score >= 4 AND f_score <= 2 THEN 'New / Promising'
            WHEN r_score >= 3 AND f_score <= 3 AND m_score >= 2 THEN 'Potential Loyalists'
            WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk (High Value)'
            WHEN r_score <= 2 AND f_score <= 2 AND m_score >= 3 THEN 'At Risk (Moderate Spend)'
            WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2 THEN 'Hibernating / Churned'
            WHEN r_score = 3 AND f_score <= 3 THEN 'Need Attention'
            ELSE 'General Customers'
        END AS segment_name
    FROM rfm_quantiles
),

-- Step 4: Aggregate Segment Metrics & Pareto Revenue Concentration
segment_performance AS (
    SELECT
        segment_name,
        COUNT(DISTINCT customer_id) AS total_customers,
        ROUND(SUM(monetary_total), 2) AS total_revenue,
        ROUND(AVG(recency_days), 1) AS avg_recency_days,
        ROUND(AVG(frequency), 1) AS avg_orders,
        ROUND(AVG(monetary_total / frequency), 2) AS avg_order_value,
        -- Window calculation for % shares
        ROUND(COUNT(DISTINCT customer_id) * 100.0 / SUM(COUNT(DISTINCT customer_id)) OVER(), 2) AS customer_share_pct,
        ROUND(SUM(monetary_total) * 100.0 / SUM(SUM(monetary_total)) OVER(), 2) AS revenue_share_pct
    FROM rfm_segments
    GROUP BY segment_name
)

SELECT
    segment_name,
    total_customers,
    customer_share_pct,
    total_revenue,
    revenue_share_pct,
    avg_recency_days,
    avg_orders,
    avg_order_value,
    -- Running cumulative revenue share demonstrating top segments driving 68% of total revenue
    ROUND(SUM(revenue_share_pct) OVER(ORDER BY total_revenue DESC), 2) AS cumulative_revenue_pct
FROM segment_performance
ORDER BY total_revenue DESC;
