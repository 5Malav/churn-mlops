"""Pandera schema for churn dataset validation."""

import pandera.pandas as pa

# ============================================================
# ALLOWED VALUES (from inspection of training data)
# ============================================================

GENDER_VALUES = ["Male", "Female"]
YES_NO_VALUES = ["Yes", "No"]
INTERNET_SERVICE_VALUES = ["DSL", "Fiber optic", "No"]
INTERNET_RELATED_VALUES = ["Yes", "No", "No internet service"]
MULTIPLE_LINES_VALUES = ["Yes", "No", "No phone service"]
CONTRACT_VALUES = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD_VALUES = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

# ============================================================
# CHURN DATAFRAME SCHEMA
# ============================================================
# This is the data contract. ANY DataFrame entering our pipeline
# must match this schema, or validation fails loudly.

ChurnSchema = pa.DataFrameSchema(
    columns={
        # ---------------- IDs ----------------
        "id": pa.Column(int, pa.Check.ge(0), unique=True),
        # ---------------- Demographics ----------------
        "gender": pa.Column(str, pa.Check.isin(GENDER_VALUES)),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1])),
        "Partner": pa.Column(str, pa.Check.isin(YES_NO_VALUES)),
        "Dependents": pa.Column(str, pa.Check.isin(YES_NO_VALUES)),
        # ---------------- Account info ----------------
        "tenure": pa.Column(int, pa.Check.in_range(0, 100)),
        "Contract": pa.Column(str, pa.Check.isin(CONTRACT_VALUES)),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(YES_NO_VALUES)),
        "PaymentMethod": pa.Column(str, pa.Check.isin(PAYMENT_METHOD_VALUES)),
        # ---------------- Phone services ----------------
        "PhoneService": pa.Column(str, pa.Check.isin(YES_NO_VALUES)),
        "MultipleLines": pa.Column(str, pa.Check.isin(MULTIPLE_LINES_VALUES)),
        # ---------------- Internet services ----------------
        "InternetService": pa.Column(str, pa.Check.isin(INTERNET_SERVICE_VALUES)),
        "OnlineSecurity": pa.Column(str, pa.Check.isin(INTERNET_RELATED_VALUES)),
        "OnlineBackup": pa.Column(str, pa.Check.isin(INTERNET_RELATED_VALUES)),
        "DeviceProtection": pa.Column(str, pa.Check.isin(INTERNET_RELATED_VALUES)),
        "TechSupport": pa.Column(str, pa.Check.isin(INTERNET_RELATED_VALUES)),
        "StreamingTV": pa.Column(str, pa.Check.isin(INTERNET_RELATED_VALUES)),
        "StreamingMovies": pa.Column(str, pa.Check.isin(INTERNET_RELATED_VALUES)),
        # ---------------- Charges ----------------
        "MonthlyCharges": pa.Column(float, pa.Check.in_range(0.0, 200.0)),
        "TotalCharges": pa.Column(
            float,
            pa.Check.ge(0.0),
            nullable=True,  # allows NaN values (real-world data)
        ),
        # ---------------- Target ----------------
        "Churn": pa.Column(str, pa.Check.isin(YES_NO_VALUES)),
    },
    strict=True,  # extra columns = error
    coerce=False,  # don't auto-convert types (catch real bugs)
    ordered=False,  # column order doesn't matter
)


# ============================================================
# HELPERS
# ============================================================


def get_schema_summary() -> dict:
    """Return a summary of the schema for logging/debugging."""
    return {
        "n_columns": len(ChurnSchema.columns),
        "columns": list(ChurnSchema.columns.keys()),
        "strict_mode": ChurnSchema.strict,
        "target_column": "Churn",
    }
