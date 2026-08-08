"""
RFM (Recency, Frequency, Monetary) Customer Segmentation Engine.
Computes quantile-based customer scoring, segment categorization,
and revenue concentration metrics (e.g. top segments driving ~68% of revenue).
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_rfm_metrics(
    df: pd.DataFrame,
    reference_date: datetime = None,
) -> pd.DataFrame:
    """
    Computes base RFM values per CustomerID:
    - Recency: Days since customer's last purchase.
    - Frequency: Count of unique purchase invoices.
    - Monetary: Total monetary spend (£).
    - AvgOrderValue: Monetary / Frequency.
    - PurchaseLifespanDays: Days between first and last purchase.
    """
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: (reference_date - dates.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        Country=("Country", lambda c: c.mode()[0] if not c.empty else "Unknown"),
        TotalItems=("Quantity", "sum"),
    ).reset_index()

    rfm["PurchaseLifespanDays"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days
    rfm["AvgOrderValue"] = np.round(rfm["Monetary"] / np.maximum(rfm["Frequency"], 1), 2)
    rfm["Monetary"] = np.round(rfm["Monetary"], 2)

    return rfm


def score_rfm_quantiles(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns 1-5 scores for Recency, Frequency, and Monetary metrics:
    - Higher Recency score = more recent (lower days)
    - Higher Frequency score = more orders
    - Higher Monetary score = higher total spend
    """
    df = rfm_df.copy()

    # Recency: lower is better -> invert rank
    df["R_Score"] = pd.qcut(
        df["Recency"].rank(method="first", ascending=False),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    # Frequency: rank based qcut
    df["F_Score"] = pd.qcut(
        df["Frequency"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    # Monetary: rank based qcut
    df["M_Score"] = pd.qcut(
        df["Monetary"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    df["RFM_Score"] = (
        df["R_Score"].astype(str)
        + df["F_Score"].astype(str)
        + df["M_Score"].astype(str)
    )
    df["RFM_Sum"] = df["R_Score"] + df["F_Score"] + df["M_Score"]

    return df


def assign_rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps customer RFM scores to strategic behavioral segments.
    """
    scored = df.copy()

    def segment_mapper(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3 and m >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New / Promising"
        elif r >= 3 and f <= 3 and m >= 2:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3:
            return "At Risk (High Value)"
        elif r <= 2 and f <= 2 and m >= 3:
            return "At Risk (Moderate Spend)"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Hibernating / Churned"
        elif r == 3 and f <= 3:
            return "Need Attention"
        else:
            return "General Customers"

    scored["Segment"] = scored.apply(segment_mapper, axis=1)
    return scored


def calculate_segment_insights(rfm_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Aggregates metrics by segment to analyze customer distribution,
    total revenue contribution, and pareto concentration.
    """
    total_rev = rfm_df["Monetary"].sum()
    total_cust = len(rfm_df)

    summary = rfm_df.groupby("Segment").agg(
        CustomerCount=("CustomerID", "count"),
        TotalRevenue=("Monetary", "sum"),
        AvgRecencyDays=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetarySpend=("Monetary", "mean"),
        AvgOrderValue=("AvgOrderValue", "mean"),
    ).reset_index()

    summary["CustomerSharePct"] = np.round((summary["CustomerCount"] / total_cust) * 100, 2)
    summary["RevenueSharePct"] = np.round((summary["TotalRevenue"] / total_rev) * 100, 2)
    summary["TotalRevenue"] = np.round(summary["TotalRevenue"], 2)
    summary["AvgRecencyDays"] = np.round(summary["AvgRecencyDays"], 1)
    summary["AvgFrequency"] = np.round(summary["AvgFrequency"], 1)
    summary["AvgMonetarySpend"] = np.round(summary["AvgMonetarySpend"], 2)
    summary["AvgOrderValue"] = np.round(summary["AvgOrderValue"], 2)

    summary = summary.sort_values(by="TotalRevenue", ascending=False).reset_index(drop=True)

    # Top segments cumulative revenue
    top_2_revenue_pct = float(summary.iloc[:2]["RevenueSharePct"].sum()) if len(summary) >= 2 else 0.0

    insights = {
        "total_customers": total_cust,
        "total_revenue": round(float(total_rev), 2),
        "top_segments_revenue_pct": round(top_2_revenue_pct, 2),
        "champions_revenue_pct": round(
            float(summary.loc[summary["Segment"] == "Champions", "RevenueSharePct"].sum()), 2
        ),
        "at_risk_revenue": round(
            float(summary.loc[summary["Segment"].str.contains("At Risk"), "TotalRevenue"].sum()), 2
        ),
    }

    return summary, insights


def run_rfm_pipeline(clean_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    End-to-end execution of RFM scoring and segmentation.
    """
    base_rfm = compute_rfm_metrics(clean_df)
    scored_rfm = score_rfm_quantiles(base_rfm)
    segmented_rfm = assign_rfm_segments(scored_rfm)
    summary_df, insights = calculate_segment_insights(segmented_rfm)
    return segmented_rfm, summary_df, insights


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.cleaner import clean_retail_data

    raw = load_raw_data()
    clean, _ = clean_retail_data(raw)
    rfm_df, summary, metrics = run_rfm_pipeline(clean)
    print("=== RFM Segmentation Insights ===")
    print(metrics)
    print("\n=== Segment Performance Summary Table ===")
    print(summary.to_string(index=False))
