-- ====================================================================
-- 02_COHORT_ANALYSIS.SQL
-- Customer Acquisition Month Cohort Retention Matrix
-- Calculates month-over-month customer retention percentages
-- ====================================================================

WITH cleaned_data AS (
    SELECT
        CustomerID AS customer_id,
        InvoiceNo AS invoice_no,
        InvoiceDate AS invoice_date,
        TotalAmount AS total_amount,
        DATE_TRUNC(DATE(InvoiceDate), MONTH) AS invoice_month
    FROM `ecommerce_db.online_retail_cleaned`
),

-- Step 1: Find first acquisition month for each customer
customer_cohorts AS (
    SELECT
        customer_id,
        MIN(invoice_month) AS cohort_month
    FROM cleaned_data
    GROUP BY customer_id
),

-- Step 2: Combine transactions with cohort month
cohort_activities AS (
    SELECT
        t.customer_id,
        c.cohort_month,
        t.invoice_month,
        -- Calculate number of months elapsed since acquisition (0, 1, 2, 3...)
        DATE_DIFF(t.invoice_month, c.cohort_month, MONTH) AS cohort_index,
        t.total_amount
    FROM cleaned_data t
    JOIN customer_cohorts c ON t.customer_id = c.customer_id
),

-- Step 3: Count active unique customers per cohort index
cohort_counts AS (
    SELECT
        cohort_month,
        cohort_index,
        COUNT(DISTINCT customer_id) AS active_customers,
        ROUND(SUM(total_amount), 2) AS cohort_revenue
    FROM cohort_activities
    GROUP BY cohort_month, cohort_index
),

-- Step 4: Compute cohort base size (Month 0) and retention percentage
cohort_retention AS (
    SELECT
        cohort_month,
        cohort_index,
        active_customers,
        cohort_revenue,
        -- First value in window is month 0 cohort base
        FIRST_VALUE(active_customers) OVER(
            PARTITION BY cohort_month
            ORDER BY cohort_index
        ) AS cohort_base_size,
        ROUND(
            (active_customers * 100.0) / FIRST_VALUE(active_customers) OVER(
                PARTITION BY cohort_month
                ORDER BY cohort_index
            ),
            2
        ) AS retention_rate_pct
    FROM cohort_counts
)

-- Final Output: Cohort Retention Table
SELECT
    cohort_month,
    cohort_index,
    cohort_base_size,
    active_customers,
    retention_rate_pct,
    cohort_revenue
FROM cohort_retention
ORDER BY cohort_month, cohort_index;
