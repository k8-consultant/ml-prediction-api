
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest

from app.model import predict


app = FastAPI(
    title="ML Prediction API",
    version="2.0.0",
)


# --------------------------------------------------
# Application / Model Version
# --------------------------------------------------

APP_VERSION = os.getenv("APP_VERSION", "v2")
MODEL_VERSION = os.getenv("MODEL_VERSION", "model-v1")


# --------------------------------------------------
# Prometheus Metrics
# --------------------------------------------------

REQUEST_COUNT = Counter(
    "ml_api_requests_total",
    "Total number of prediction API requests",
    ["endpoint", "status"],
)


REQUEST_LATENCY = Histogram(
    "ml_api_request_latency_seconds",
    "Prediction API request latency in seconds",
    ["endpoint"],
)


# --------------------------------------------------
# Request Model
# --------------------------------------------------

class PredictionRequest(BaseModel):
    age: int
    income: float
    credit_score: int


# --------------------------------------------------
# Health Endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application_version": APP_VERSION,
        "model_version": MODEL_VERSION,
    }


# --------------------------------------------------
# Version Endpoint
# --------------------------------------------------

@app.get("/version")
def version():
    return {
        "application_version": APP_VERSION,
        "model_version": MODEL_VERSION,
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def make_prediction(data: PredictionRequest):

    start_time = time.time()

    try:

        # --------------------------------------------------
        # Intentional V2 failure for Canary testing
        # --------------------------------------------------

        if APP_VERSION == "v2":

            REQUEST_COUNT.labels(
                endpoint="/predict",
                status="error",
            ).inc()

            raise HTTPException(
                status_code=500,
                detail="Intentional V2 canary failure",
            )

        # --------------------------------------------------
        # Normal prediction behaviour
        # --------------------------------------------------

        result = predict(
            age=data.age,
            income=data.income,
            credit_score=data.credit_score,
        )

        REQUEST_COUNT.labels(
            endpoint="/predict",
            status="success",
        ).inc()

        return {
            **result,
            "application_version": APP_VERSION,
            "model_version": MODEL_VERSION,
        }

    except HTTPException:
        # Important:
        # The error metric was already incremented above.
        raise

    except Exception as exc:

        REQUEST_COUNT.labels(
            endpoint="/predict",
            status="error",
        ).inc()

        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc

    finally:

        REQUEST_LATENCY.labels(
            endpoint="/predict",
        ).observe(
            time.time() - start_time
        )


# --------------------------------------------------
# Prometheus Metrics Endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )