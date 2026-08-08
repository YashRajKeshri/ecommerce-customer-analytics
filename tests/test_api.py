"""
Unit & Integration Tests for FastAPI REST Endpoints.
"""

from fastapi.testclient import TestClient
import pytest
from api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data


def test_predict_churn_endpoint():
    payload = {
        "CustomerID": "TEST_CUST_999",
        "RecencyDays": 80.0,
        "Frequency": 2,
        "TotalSpend": 450.0,
        "AvgOrderValue": 225.0,
        "TotalQuantity": 40,
        "AvgItemsPerOrder": 20.0,
        "TenureDays": 120.0,
        "AvgOrderIntervalDays": 60.0,
        "StdOrderIntervalDays": 5.0,
        "SpendVelocityLast30d": 0.05,
        "UniqueProducts": 8,
        "IsUK": 1,
    }
    response = client.post("/api/v1/predict-churn", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert 0.0 <= res_data["churn_probability"] <= 1.0
    assert "risk_tier" in res_data
    assert "recommended_action" in res_data


def test_batch_predict_endpoint():
    payload = {
        "customers": [
            {
                "CustomerID": "BATCH_1",
                "RecencyDays": 10.0,
                "Frequency": 10,
                "TotalSpend": 3000.0,
                "AvgOrderValue": 300.0,
                "TotalQuantity": 300,
                "AvgItemsPerOrder": 30.0,
                "TenureDays": 300.0,
                "AvgOrderIntervalDays": 30.0,
                "StdOrderIntervalDays": 4.0,
                "SpendVelocityLast30d": 0.40,
                "UniqueProducts": 25,
                "IsUK": 1,
            },
            {
                "CustomerID": "BATCH_2",
                "RecencyDays": 110.0,
                "Frequency": 1,
                "TotalSpend": 50.0,
                "AvgOrderValue": 50.0,
                "TotalQuantity": 5,
                "AvgItemsPerOrder": 5.0,
                "TenureDays": 110.0,
                "AvgOrderIntervalDays": 110.0,
                "StdOrderIntervalDays": 0.0,
                "SpendVelocityLast30d": 0.0,
                "UniqueProducts": 2,
                "IsUK": 0,
            },
        ]
    }
    response = client.post("/api/v1/batch-predict-churn", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["total_evaluated"] == 2
    assert len(res_data["predictions"]) == 2
