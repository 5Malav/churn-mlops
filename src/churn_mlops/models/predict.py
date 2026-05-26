"""Load the trained model and make churn predictions for one customer."""

from pathlib import Path

import lightgbm as lgb
import pandas as pd

# Where the trained model lives (created in Phase 3)
MODEL_PATH = Path("models/churn_model.txt")

# Text columns the model expects as "category" type
CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_model() -> lgb.Booster:
    """Read the trained model file from disk."""
    return lgb.Booster(model_file=str(MODEL_PATH))


def predict_churn(model: lgb.Booster, customer: dict) -> dict:
    """Take one customer (a dict), return churn probability + prediction."""
    # 1. Turn the dict into a one-row DataFrame
    df = pd.DataFrame([customer])

    # 2. Set the text columns to 'category' type (what the model expects)
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    # 3. Predict — model gives a probability between 0 and 1
    probability = model.predict(df)[0]

    # 4. Turn probability into a yes/no (0.5 is the cutoff)
    prediction = int(probability >= 0.5)

    return {
        "churn_probability": round(float(probability), 4),
        "churn_prediction": prediction,
    }
