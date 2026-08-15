import os

from fastapi import FastAPI
from pydantic import BaseModel

from app.model import predict


APP_VERSION = os.getenv("APP_VERSION", "v1")
MODEL_VERSION = os.getenv("MODEL_VERSION", "model-v1")


app = FastAPI(
    title="ML Prediction API",
    version=APP_VERSION,
)


class PredictionRequest(BaseModel):
    age: int
    income: float
    credit_score: int


@app.get("/")
def root():
    return {
        "service": "ml-prediction-api",
        "version": APP_VERSION,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": APP_VERSION,
    }


@app.get("/version")
def version():
    return {
        "application_version": APP_VERSION,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
def make_prediction(data: PredictionRequest):

    result = predict(
        age=data.age,
        income=data.income,
        credit_score=data.credit_score,
    )

    return {
        **result,
        "application_version": APP_VERSION,
        "model_version": MODEL_VERSION,
    }

