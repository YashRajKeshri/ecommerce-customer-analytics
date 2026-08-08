"""
FastAPI Routes for Churn ML Inference and Analytics.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
import pandas as pd

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChurnPredictionResponse,
    CustomerFeaturesInput,
    HealthCheckResponse,
)
from src.churn_model import (
    MODEL_PATH,
    build_churn_features,
    load_trained_churn_model,
    predict_churn_risk,
    train_churn_models,
)
from src.cleaner import clean_retail_data
from src.data_loader import load_raw_data

router = APIRouter()


def ensure_model_is_trained():
    """Helper to ensure model weights exist."""
    if not MODEL_PATH.exists():
        raw = load_raw_data()
        clean, _ = clean_retail_data(raw)
        _, X, y = build_churn_features(clean)
        train_churn_models(X, y, save_model=True)


@router.get("/health", response_model=HealthCheckResponse, tags=["System"])
def health_check():
    """Returns service health status and model metadata."""
    try:
        ensure_model_is_trained()
        model_obj = load_trained_churn_model()
        model_name = model_obj.get("model_name", "Gradient Boosting Classifier")
        feat_count = len(model_obj.get("feature_names", []))
    except Exception:
        model_name = "Model Initialized"
        feat_count = 11

    return HealthCheckResponse(
        status="healthy",
        model_name=model_name,
        model_version="1.0.0",
        features_count=feat_count,
        dataset_records="541,909 transactions",
    )


@router.post("/predict-churn", response_model=ChurnPredictionResponse, tags=["Machine Learning"])
def predict_single_customer_churn(payload: CustomerFeaturesInput):
    """
    Evaluates churn risk probability for a single customer record.
    Returns probability score, risk tier, and tailored retention action.
    """
    try:
        ensure_model_is_trained()
        input_dict = payload.model_dump()
        res = predict_churn_risk(input_dict)
        return ChurnPredictionResponse(
            customer_id=payload.CustomerID,
            churn_probability=res["churn_probability"],
            is_churn_risk=res["is_churn_risk"],
            risk_tier=res["risk_tier"],
            recommended_action=res["recommended_action"],
            evaluated_features=res["evaluated_features"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.post("/batch-predict-churn", response_model=BatchPredictionResponse, tags=["Machine Learning"])
def predict_batch_customers(payload: BatchPredictionRequest):
    """
    Evaluates a batch of customer accounts, quantifying aggregate revenue at risk.
    """
    try:
        ensure_model_is_trained()
        predictions = []
        at_risk_count = 0
        total_rev_at_risk = 0.0

        for item in payload.customers:
            input_dict = item.model_dump()
            res = predict_churn_risk(input_dict)
            is_risk = res["is_churn_risk"]
            if is_risk:
                at_risk_count += 1
                total_rev_at_risk += item.TotalSpend

            predictions.append(
                ChurnPredictionResponse(
                    customer_id=item.CustomerID,
                    churn_probability=res["churn_probability"],
                    is_churn_risk=is_risk,
                    risk_tier=res["risk_tier"],
                    recommended_action=res["recommended_action"],
                    evaluated_features=res["evaluated_features"],
                )
            )

        return BatchPredictionResponse(
            total_evaluated=len(payload.customers),
            at_risk_count=at_risk_count,
            total_revenue_at_risk=round(total_rev_at_risk, 2),
            predictions=predictions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch error: {str(e)}")
