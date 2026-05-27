"""FastAPI app — serves churn predictions over HTTP."""

from fastapi import FastAPI, HTTPException
from loguru import logger

from src.churn_mlops.api.schemas import CustomerFeatures, PredictionResponse
from src.churn_mlops.models.predict import load_model, predict_churn

# Create the app
app = FastAPI(title="Churn Prediction API")

# Load the model ONCE when the app starts (not on every request)
model = load_model()


@app.get("/")
def home() -> dict:
    """Welcome message."""
    return {"message": "Churn Prediction API. See /docs for usage."}


@app.get("/health")
def health() -> dict:
    """Health check — is the API alive?"""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> dict:
    """Take a customer, return churn probability + prediction."""
    try:
        customer_dict = customer.model_dump()
        return predict_churn(model, customer_dict)
    except Exception as e:
        # Log the REAL error for us to debug (server-side only)
        logger.error(f"Prediction failed: {e}")
        # Return a CLEAN message to the user (no internals leaked)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please check your input and try again.",
        ) from e
