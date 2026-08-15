
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_version():
    response = client.get("/version")

    assert response.status_code == 200
    assert "application_version" in response.json()


def test_prediction():
    payload = {
        "age": 22,
        "income": 25000,
        "credit_score": 550,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "model_version" in result