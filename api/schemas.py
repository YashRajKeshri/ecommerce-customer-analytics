"""
Pydantic Schemas for FastAPI REST API endpoints.
Validates input features, churn risk responses, and analytics payloads.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CustomerFeaturesInput(BaseModel):
    CustomerID: Optional[str] = Field(default="12345", description="Unique Customer Identifier")
    RecencyDays: float = Field(..., ge=0, description="Days elapsed since customer's last purchase")
    Frequency: int = Field(..., ge=1, description="Total number of distinct purchase invoices")
    TotalSpend: float = Field(..., ge=0, description="Total cumulative spend (£)")
    AvgOrderValue: float = Field(..., ge=0, description="Average spend per purchase order (£)")
    TotalQuantity: int = Field(..., ge=1, description="Total items purchased")
    AvgItemsPerOrder: float = Field(..., ge=0, description="Average items per order")
    TenureDays: float = Field(..., ge=0, description="Customer relationship lifespan in days")
    AvgOrderIntervalDays: float = Field(..., ge=0, description="Average days between consecutive orders")
    StdOrderIntervalDays: float = Field(..., ge=0, description="Standard deviation of order intervals")
    SpendVelocityLast30d: float = Field(..., ge=0.0, le=1.0, description="Proportion of total spend within last 30 days")
    UniqueProducts: int = Field(..., ge=1, description="Count of distinct product stock codes bought")
    IsUK: int = Field(..., ge=0, le=1, description="1 if United Kingdom, 0 if International")


class ChurnPredictionResponse(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    is_churn_risk: bool
    risk_tier: str
    recommended_action: str
    evaluated_features: Dict[str, Any]


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeaturesInput]


class BatchPredictionResponse(BaseModel):
    total_evaluated: int
    at_risk_count: int
    total_revenue_at_risk: float
    predictions: List[ChurnPredictionResponse]


class HealthCheckResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    features_count: int
    dataset_records: str
