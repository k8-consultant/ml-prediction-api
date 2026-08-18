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

APP_VERSION = os.getenv("APP_VERSION", "v2")
MODEL_VERSION = os.getenv("MODEL_VERSION", "model-v1")
FAIL_CANARY = os.getenv("FAIL_CANARY", "false").lower() == "false"

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


class PredictionRequest(BaseModel):
    age: int
    income: float
    credit_score: int


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application_version": APP_VERSION,
        "model_version": MODEL_VERSION,
    }


@app.get("/version")
def version():
    return {
        "application_version": APP_VERSION,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
def make_prediction(data: PredictionRequest):
    start_time = time.time()

    try:
        if FAIL_CANARY:
            REQUEST_COUNT.labels(
                endpoint="/predict",
                status="error",
            ).inc()

            raise HTTPException(
                status_code=500,
                detail="Intentional canary runtime failure",
            )

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
            "prediction": result["prediction"],
            "risk_score": result["risk_score"],
            "application_version": APP_VERSION,
            "model_version": MODEL_VERSION,
        }

    except HTTPException:
        raise

    except Exception:
        REQUEST_COUNT.labels(
            endpoint="/predict",
            status="error",
        ).inc()

        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        )

    finally:
        REQUEST_LATENCY.labels(
            endpoint="/predict",
        ).observe(time.time() - start_time)


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )