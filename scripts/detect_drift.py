"""Data drift detection using Evidently.

Compares a reference dataset (what the model was trained on) against a
current dataset (new incoming data) and generates an HTML drift report.

For this demo, 'current' data is simulated by shifting a few features of a
data slice — showing that the detection catches distribution changes. In
production, 'current' would be real incoming prediction data.
"""

from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset

from src.churn_mlops.data.loader import load_churn_data

# Feature columns the model uses (exclude the target + any ID)
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
]

OUTPUT_PATH = Path("reports/drift_report.html")


def main() -> None:
    # Load the full dataset
    df = load_churn_data(Path("data/raw/train.csv"))
    df = df[FEATURE_COLUMNS]

    # Split into two halves
    midpoint = len(df) // 2
    reference = df.iloc[:midpoint].copy()
    current = df.iloc[midpoint:].copy()

    # Simulate drift in the 'current' data: shift a few features
    # (imagine customers now have longer tenure + higher charges)
    current["tenure"] = current["tenure"] + 20
    current["MonthlyCharges"] = current["MonthlyCharges"] * 1.3

    # Build and run the drift report
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    # Save the HTML report
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(OUTPUT_PATH))
    print(f"✅ Drift report saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
