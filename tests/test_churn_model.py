"""
Unit Tests for Machine Learning Churn Pipeline.
"""

import pandas as pd
import pytest
from src.churn_model import (
    build_churn_features,
    train_churn_models,
    predict_churn_risk,
    MODEL_PATH,
)
from src.data_loader import generate_synthetic_online_retail
from src.cleaner import clean_retail_data


def test_churn_training_and_inference():
    # Generate test subset
    raw = generate_synthetic_online_retail(n_records=8000, seed=42, save_path=None)
    clean, _ = clean_retail_data(raw)
    feat_df, X, y = build_churn_features(clean, observation_window_days=60)

    assert len(feat_df) > 50
    assert "Churn" in feat_df.columns
    assert set(y.unique()).issubset({0, 1})

    res = train_churn_models(X, y, save_model=True)

    assert res["champion_roc_auc"] >= 0.70
    assert MODEL_PATH.exists()

    # Test single customer prediction
    test_customer = {
        "RecencyDays": 95.0,
        "Frequency": 1,
        "TotalSpend": 85.0,
        "AvgOrderValue": 85.0,
        "TotalQuantity": 10,
        "AvgItemsPerOrder": 10.0,
        "TenureDays": 95.0,
        "AvgOrderIntervalDays": 95.0,
        "StdOrderIntervalDays": 0.0,
        "SpendVelocityLast30d": 0.0,
        "UniqueProducts": 3,
        "IsUK": 1,
    }

    pred = predict_churn_risk(test_customer)
    assert 0.0 <= pred["churn_probability"] <= 1.0
    assert "risk_tier" in pred
    assert "recommended_action" in pred
