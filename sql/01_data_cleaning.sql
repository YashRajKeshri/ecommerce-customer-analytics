-- ====================================================================
-- 01_DATA_CLEANING.SQL
-- E-Commerce Customer Analytics Data Cleaning & Standardization Script
-- Standardizes 541,000+ transaction records for analytical reporting
-- ====================================================================

-- CTE: Remove cancelled orders ('C' prefix), negative quantities, and null customers
WITH raw_filtered AS (
    SELECT
        TRIM(InvoiceNo) AS invoice_no,
        TRIM(StockCode) AS stock_code,
        COALESCE(TRIM(Description), 'UNKNOWN') AS description,
        CAST(Quantity AS INT64) AS quantity,
        CAST(InvoiceDate AS TIMESTAMP) AS invoice_date,
        CAST(UnitPrice AS NUMERIC) AS unit_price,
        CAST(CustomerID AS STRING) AS customer_id,
        TRIM(Country) AS country
    FROM `ecommerce_db.online_retail_raw`
    WHERE
        -- Filter out cancellation invoices starting with 'C'
        InvoiceNo NOT LIKE 'C%'
        -- Filter out return transactions and non-positive prices
        AND Quantity > 0
        AND UnitPrice > 0.00
        -- Retain identified customer accounts for segmentation
        AND CustomerID IS NOT NULL
),

-- CTE: Calculate total amount and temporal dimension columns
cleaned_transactions AS (
    SELECT
        invoice_no,
        stock_code,
        description,
        quantity,
        invoice_date,
        unit_price,
        ROUND(quantity * unit_price, 2) AS total_amount,
        customer_id,
        country,
        -- Date dimensions
        EXTRACT(YEAR FROM invoice_date) AS invoice_year,
        EXTRACT(MONTH FROM invoice_date) AS invoice_month,
        FORMAT_TIMESTAMP('%Y-%m', invoice_date) AS invoice_year_month,
        FORMAT_TIMESTAMP('%A', invoice_date) AS day_of_week,
        EXTRACT(HOUR FROM invoice_date) AS invoice_hour
    FROM raw_filtered
)

-- Materialize clean table or export
SELECT * FROM cleaned_transactions;
