"""
Data Cleaning & Transformation Pipeline for E-Commerce Transactions.
Applies rigorous data validation, cancellation handling, outlier removal,
and feature preparation for RFM, Cohort, and ML modeling.
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


def clean_retail_data(
    df: pd.DataFrame,
    drop_missing_customers: bool = True,
    remove_cancellations: bool = True,
    price_upper_quantile: float = 0.999,
    quantity_upper_quantile: float = 0.999,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw retail transaction data:
    1. Standardizes datatypes and column names.
    2. Drops or isolates missing CustomerID records (~135k guest transactions).
    3. Handles cancellations (invoices starting with 'C' and negative quantities).
    4. Filters non-positive unit prices and extreme statistical anomalies.
    5. Calculates TotalAmount = Quantity * UnitPrice.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned dataframe and metadata metrics.
    """
    initial_rows = len(df)
    initial_revenue = (
        (df["Quantity"] * df["UnitPrice"]).sum()
        if "Quantity" in df.columns and "UnitPrice" in df.columns
        else 0.0
    )

    clean_df = df.copy()

    # 1. Standardize types
    clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"])
    clean_df["InvoiceNo"] = clean_df["InvoiceNo"].astype(str).str.strip()
    clean_df["StockCode"] = clean_df["StockCode"].astype(str).str.strip()
    if "Description" in clean_df.columns:
        clean_df["Description"] = clean_df["Description"].fillna("UNKNOWN").astype(str).str.strip()

    # 2. Track cancellations
    is_cancelled_invoice = clean_df["InvoiceNo"].str.startswith("C", na=False)
    is_negative_qty = clean_df["Quantity"] <= 0
    cancellation_mask = is_cancelled_invoice | is_negative_qty
    n_cancellations = int(cancellation_mask.sum())

    if remove_cancellations:
        clean_df = clean_df[~cancellation_mask].copy()

    # 3. Missing customer handling
    missing_cust_mask = clean_df["CustomerID"].isnull()
    n_missing_customers = int(missing_cust_mask.sum())

    if drop_missing_customers:
        clean_df = clean_df[~missing_cust_mask].copy()
        clean_df["CustomerID"] = clean_df["CustomerID"].astype(int).astype(str)

    # 4. Filter invalid or extreme prices and quantities
    price_mask = clean_df["UnitPrice"] > 0
    clean_df = clean_df[price_mask].copy()

    # Filter extreme outliers based on quantiles when dataframe is large
    n_outliers = 0
    if len(clean_df) >= 100:
        q_price = clean_df["UnitPrice"].quantile(price_upper_quantile)
        q_qty = clean_df["Quantity"].quantile(quantity_upper_quantile)
        outlier_mask = (clean_df["UnitPrice"] <= q_price) & (clean_df["Quantity"] <= q_qty)
        n_outliers = int((~outlier_mask).sum())
        clean_df = clean_df[outlier_mask].copy()

    # 5. Compute revenue / TotalAmount
    clean_df["TotalAmount"] = np.round(clean_df["Quantity"] * clean_df["UnitPrice"], 2)

    # 6. Extract temporal features
    clean_df["InvoiceYear"] = clean_df["InvoiceDate"].dt.year
    clean_df["InvoiceMonth"] = clean_df["InvoiceDate"].dt.month
    clean_df["InvoiceYearMonth"] = clean_df["InvoiceDate"].dt.to_period("M").astype(str)
    clean_df["DayOfWeek"] = clean_df["InvoiceDate"].dt.day_name()
    clean_df["Hour"] = clean_df["InvoiceDate"].dt.hour

    final_rows = len(clean_df)
    final_revenue = float(clean_df["TotalAmount"].sum())
    unique_customers = int(clean_df["CustomerID"].nunique()) if "CustomerID" in clean_df.columns else 0
    unique_invoices = int(clean_df["InvoiceNo"].nunique())

    metrics = {
        "initial_rows": initial_rows,
        "initial_revenue": round(float(initial_revenue), 2),
        "n_cancellations_filtered": n_cancellations,
        "n_missing_customers_filtered": n_missing_customers,
        "n_outliers_filtered": n_outliers,
        "final_rows": final_rows,
        "final_revenue": round(final_revenue, 2),
        "unique_customers": unique_customers,
        "unique_invoices": unique_invoices,
        "date_range_start": str(clean_df["InvoiceDate"].min()) if not clean_df.empty else "",
        "date_range_end": str(clean_df["InvoiceDate"].max()) if not clean_df.empty else "",
    }

    return clean_df, metrics
