"""Tests for the churn prediction API."""

from fastapi.testclient import TestClient

from src.churn_mlops.api.app import app

client = TestClient(app)

# A valid customer (reused across tests)
VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.70,
    "TotalCharges": 151.65,
}


def test_health_works():
    """Health check returns 200 and 'healthy'."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_works():
    """Valid customer returns a prediction with the right shape."""
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert "churn_prediction" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)


def test_bad_input_rejected():
    """Broken customer (missing fields) returns 422."""
    response = client.post("/predict", json={"gender": "Female"})
    assert response.status_code == 422


def test_prediction_error_handled(monkeypatch):
    """If prediction crashes internally, API returns a clean 500 (not a traceback)."""

    # Replace predict_churn with a function that always crashes
    def boom(*args, **kwargs):
        raise ValueError("simulated model failure")

    monkeypatch.setattr("src.churn_mlops.api.app.predict_churn", boom)

    response = client.post("/predict", json=VALID_CUSTOMER)

    # API should catch the crash and return a clean 500
    assert response.status_code == 500
    assert response.json() == {
        "detail": "Prediction failed. Please check your input and try again."
    }
