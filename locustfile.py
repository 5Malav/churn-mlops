"""Load test for the churn prediction API.

Simulates users repeatedly hitting /health and /predict to measure
throughput, latency, and failure rate under load.

Run:  locust -f locustfile.py --host http://localhost:8000
Then open http://localhost:8089 to control the test.
"""

from locust import HttpUser, between, task

# A realistic high-churn customer payload
SAMPLE_CUSTOMER = {
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


class ChurnAPIUser(HttpUser):
    """A simulated user hitting the churn API."""

    # Each user waits 1-3 seconds between requests (realistic pacing)
    wait_time = between(1, 3)

    @task(3)
    def predict(self) -> None:
        """Hit /predict (weighted 3x — the main endpoint)."""
        self.client.post("/predict", json=SAMPLE_CUSTOMER)

    @task(1)
    def health(self) -> None:
        """Hit /health (weighted 1x)."""
        self.client.get("/health")
