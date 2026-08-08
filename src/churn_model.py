"""
Predictive Churn Machine Learning Pipeline.
Implements time-split feature engineering, model training (Random Forest, Gradient Boosting),
hyperparameter tuning, ROC-AUC / F1 evaluation, and production model serialization.
"""

from datetime import datetime, timedelta
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"


def build_churn_features(
    clean_df: pd.DataFrame,
    observation_window_days: int = 90,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Builds temporal features up to cutoff date and computes ground-truth churn
    based on the subsequent observation window.

    A customer is labeled churned (1) if they made 0 purchases in the subsequent window,
    and retained (0) if they purchased during the subsequent window.
    """
    df = clean_df.copy()
    max_date = df["InvoiceDate"].max()
    cutoff_date = max_date - timedelta(days=observation_window_days)

    # Feature window data (before cutoff)
    hist_df = df[df["InvoiceDate"] <= cutoff_date].copy()
    # Target window data (after cutoff)
    future_df = df[df["InvoiceDate"] > cutoff_date].copy()

    # Active customers in future window
    retained_customers = set(future_df["CustomerID"].dropna().unique())

    # Build features for each customer in hist_df
    customer_features = []
    for cust_id, group in hist_df.groupby("CustomerID"):
        group = group.sort_values(by="InvoiceDate")
        n_orders = group["InvoiceNo"].nunique()
        total_spend = group["TotalAmount"].sum()
        total_qty = group["Quantity"].sum()
        first_date = group["InvoiceDate"].min()
        last_date = group["InvoiceDate"].max()

        recency_days = (cutoff_date - last_date).days
        tenure_days = max((cutoff_date - first_date).days, 1)

        # Order intervals
        invoice_dates = group["InvoiceDate"].drop_duplicates().sort_values()
        if len(invoice_dates) > 1:
            diffs = invoice_dates.diff().dt.days.dropna()
            avg_interval = diffs.mean()
            std_interval = diffs.std() if len(diffs) > 1 else 0.0
        else:
            avg_interval = float(tenure_days)
            std_interval = 0.0

        # Velocity: spend in last 30 days vs total spend
        last_30_spend = group[group["InvoiceDate"] >= (cutoff_date - timedelta(days=30))]["TotalAmount"].sum()
        spend_velocity = last_30_spend / max(total_spend, 1.0)

        # Country
        is_uk = 1 if group["Country"].iloc[0] == "United Kingdom" else 0
        unique_items = group["StockCode"].nunique()

        # Target: 1 if churned (NOT in retained_customers), 0 if active
        is_churned = 1 if cust_id not in retained_customers else 0

        customer_features.append({
            "CustomerID": str(cust_id),
            "RecencyDays": recency_days,
            "Frequency": n_orders,
            "TotalSpend": round(float(total_spend), 2),
            "AvgOrderValue": round(float(total_spend / max(n_orders, 1)), 2),
            "TotalQuantity": int(total_qty),
            "AvgItemsPerOrder": round(float(total_qty / max(n_orders, 1)), 1),
            "TenureDays": tenure_days,
            "AvgOrderIntervalDays": round(float(avg_interval), 1),
            "StdOrderIntervalDays": round(float(0.0 if np.isnan(std_interval) else std_interval), 1),
            "SpendVelocityLast30d": round(float(spend_velocity), 3),
            "UniqueProducts": unique_items,
            "IsUK": is_uk,
            "Churn": is_churned,
        })

    feat_df = pd.DataFrame(customer_features)
    X = feat_df.drop(columns=["CustomerID", "Churn"])
    y = feat_df["Churn"]

    return feat_df, X, y


def train_churn_models(
    X: pd.DataFrame,
    y: pd.Series,
    save_model: bool = True,
) -> Dict[str, Any]:
    """
    Trains and benchmarks Logistic Regression, Random Forest, and Gradient Boosting.
    Performs 5-Fold Stratified Cross-Validation and persists the champion model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=42, max_iter=500)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=7,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=42,
        ),
    }

    results = {}
    best_score = 0.0
    best_model_name = ""
    champion_pipeline = None

    for name, model in models.items():
        # Stratified K-Fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_rocs = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            m = model
            m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            val_probs = m.predict_proba(X_train.iloc[val_idx])[:, 1]
            cv_rocs.append(roc_auc_score(y_train.iloc[val_idx], val_probs))

        # Final fit on full train set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = float(roc_auc_score(y_test, y_proba))
        f1 = float(f1_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        results[name] = {
            "cv_roc_auc_mean": round(float(np.mean(cv_rocs)), 4),
            "cv_roc_auc_std": round(float(np.std(cv_rocs)), 4),
            "test_roc_auc": round(auc, 4),
            "test_f1": round(f1, 4),
            "test_precision": round(prec, 4),
            "test_recall": round(rec, 4),
            "test_accuracy": round(acc, 4),
            "confusion_matrix": cm,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

        if auc > best_score:
            best_score = auc
            best_model_name = name
            champion_pipeline = model

    # Feature importances (from tree model)
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    feat_importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": np.round(importances, 4),
    }).sort_values(by="Importance", ascending=False)

    payload = {
        "benchmark_results": results,
        "champion_model_name": best_model_name,
        "champion_roc_auc": round(best_score, 4),
        "feature_importances": feat_importance_df.to_dict(orient="records"),
        "feature_names": list(X.columns),
    }

    if save_model and champion_pipeline is not None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(
            {
                "model": champion_pipeline,
                "model_name": best_model_name,
                "feature_names": list(X.columns),
                "metrics": results[best_model_name],
            },
            MODEL_PATH,
        )
        print(f"Champion Model ({best_model_name}) saved to {MODEL_PATH}")

    return payload


def load_trained_churn_model() -> Dict[str, Any]:
    """Loads saved model artifact from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run training first.")
    return joblib.load(MODEL_PATH)


def predict_churn_risk(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inference endpoint for single-customer churn risk assessment.
    """
    model_obj = load_trained_churn_model()
    model = model_obj["model"]
    feat_names = model_obj["feature_names"]

    # Align input features
    row_dict = {f: customer_data.get(f, 0.0) for f in feat_names}
    input_df = pd.DataFrame([row_dict])

    prob = float(model.predict_proba(input_df)[0][1])

    # Risk tiers
    if prob >= 0.75:
        tier = "Critical Risk"
        action = "Immediate 20% re-engagement offer & VIP concierge outreach."
    elif prob >= 0.50:
        tier = "High Risk"
        action = "Automated win-back email sequence & product recommendation trigger."
    elif prob >= 0.25:
        tier = "Moderate Risk"
        action = "Category loyalty reward points & personalized digest."
    else:
        tier = "Low Risk / Healthy"
        action = "Cross-sell trending new arrivals & referral incentive."

    return {
        "churn_probability": round(prob, 4),
        "risk_tier": tier,
        "is_churn_risk": prob >= 0.50,
        "recommended_action": action,
        "evaluated_features": row_dict,
    }


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.cleaner import clean_retail_data

    raw = load_raw_data()
    clean, _ = clean_retail_data(raw)
    feat_df, X, y = build_churn_features(clean)
    print(f"Features dataset shape: {feat_df.shape}. Churn rate: {y.mean():.2%}")
    metrics = train_churn_models(X, y)
    print("\n=== Model Benchmark Results ===")
    for model_name, res in metrics["benchmark_results"].items():
        print(f"[{model_name}] ROC-AUC: {res['test_roc_auc']} | F1: {res['test_f1']} | Recall: {res['test_recall']}")
