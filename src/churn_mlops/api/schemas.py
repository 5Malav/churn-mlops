"""Pydantic schemas — define what API requests and responses look like."""

from pydantic import BaseModel


class CustomerFeatures(BaseModel):
    """One customer's data — what the API expects in a predict request."""

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    """What the API sends back after a prediction."""

    churn_probability: float
    churn_prediction: int
