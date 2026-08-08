"""
Unit Tests for Data Cleaning & Validation Pipeline.
"""

from datetime import datetime
import pandas as pd
import pytest
from src.cleaner import clean_retail_data


def test_clean_retail_data_cancellations():
    raw_sample = pd.DataFrame({
        "InvoiceNo": ["536365", "C536366", "536367"],
        "StockCode": ["85123A", "71053", "84406B"],
        "Description": ["WHITE HEART", "WHITE METAL", "CREAM CUPID"],
        "Quantity": [6, -1, 4],
        "InvoiceDate": [datetime(2010, 12, 1), datetime(2010, 12, 1), datetime(2010, 12, 2)],
        "UnitPrice": [2.55, 3.39, 2.75],
        "CustomerID": [17850, 17850, 13047],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom"],
    })

    cleaned, stats = clean_retail_data(raw_sample)

    # Cancellation record must be removed
    assert len(cleaned) == 2
    assert "C536366" not in cleaned["InvoiceNo"].values
    assert (cleaned["Quantity"] > 0).all()
    assert (cleaned["TotalAmount"] > 0).all()
    assert stats["n_cancellations_filtered"] == 1


def test_clean_retail_data_missing_customers():
    raw_sample = pd.DataFrame({
        "InvoiceNo": ["536365", "536368"],
        "StockCode": ["85123A", "84406B"],
        "Quantity": [6, 4],
        "InvoiceDate": [datetime(2010, 12, 1), datetime(2010, 12, 2)],
        "UnitPrice": [2.55, 2.75],
        "CustomerID": [17850, None],
        "Country": ["United Kingdom", "United Kingdom"],
    })

    cleaned, stats = clean_retail_data(raw_sample, drop_missing_customers=True)

    assert len(cleaned) == 1
    assert stats["n_missing_customers_filtered"] == 1
    assert cleaned["CustomerID"].iloc[0] == "17850"
