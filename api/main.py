"""
Production FastAPI Application Entrypoint.
E-Commerce Customer Analytics & Churn Prediction REST Microservice.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="E-Commerce Customer Analytics & Churn Prediction API",
    description="""
    Production-grade REST API powering Customer Analytics, RFM Segmentation,
    Cohort Retention Analysis, and Scikit-Learn Churn Risk Prediction.
    
    Highlights:
    - 541,000+ Transactions Processing Pipeline
    - 68% Revenue Concentration Top Customer Identification
    - £780K Churn Exposure Cohort Risk Scoring
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    return {
        "project": "E-Commerce Customer Analytics & Churn Prediction Platform",
        "status": "operational",
        "documentation": "/docs",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
