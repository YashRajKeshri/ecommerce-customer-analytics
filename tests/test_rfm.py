"""
Unit Tests for RFM Segmentation and Pareto Revenue Analysis.
"""

from datetime import datetime, timedelta
import pandas as pd
import pytest
from src.rfm import run_rfm_pipeline, compute_rfm_metrics, score_rfm_quantiles, assign_rfm_segments


def test_rfm_computation():
    ref_date = datetime(2011, 12, 10)
    data = pd.DataFrame({
        "InvoiceNo": ["1001", "1002", "1003", "1004", "1005"],
        "InvoiceDate": [
            datetime(2011, 12, 8),
            datetime(2011, 12, 5),
            datetime(2011, 10, 1),
            datetime(2011, 11, 15),
            datetime(2011, 12, 9),
        ],
        "TotalAmount": [100.0, 150.0, 50.0, 200.0, 300.0],
        "CustomerID": ["C1", "C1", "C2", "C3", "C4"],
        "Quantity": [10, 15, 5, 20, 30],
        "Country": ["United Kingdom"] * 5,
    })

    rfm = compute_rfm_metrics(data, reference_date=ref_date)
    assert len(rfm) == 4
    c1 = rfm[rfm["CustomerID"] == "C1"].iloc[0]
    assert c1["Frequency"] == 2
    assert c1["Monetary"] == 250.0
    assert c1["Recency"] == 2  # Dec 10 - Dec 8


def test_rfm_pipeline_execution():
    from src.data_loader import generate_synthetic_online_retail
    from src.cleaner import clean_retail_data

    raw = generate_synthetic_online_retail(n_records=5000, seed=123, save_path=None)
    clean, _ = clean_retail_data(raw)
    rfm_df, summary, insights = run_rfm_pipeline(clean)

    assert "Segment" in rfm_df.columns
    assert len(summary) > 0
    assert insights["top_segments_revenue_pct"] > 50.0  # Demonstrates strong revenue concentration
