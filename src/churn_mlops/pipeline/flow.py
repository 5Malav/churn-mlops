"""Prefect flow that orchestrates the churn training pipeline."""

import subprocess

from prefect import flow, task
from prefect.logging import get_run_logger


@task(retries=2, retry_delay_seconds=5)
def validate_data() -> None:
    """Run the data validation step."""
    logger = get_run_logger()
    logger.info("Starting data validation...")
    subprocess.run(
        ["python", "-m", "scripts.validate_data"],
        check=True,
    )
    logger.info("✅ Data validation passed")


@task(retries=2, retry_delay_seconds=5)
def train_model() -> None:
    """Run the model training step."""
    logger = get_run_logger()
    logger.info("Starting model training...")
    subprocess.run(
        ["python", "-m", "src.churn_mlops.models.train"],
        check=True,
    )
    logger.info("✅ Model training complete")


@flow(name="churn-training-pipeline")
def churn_pipeline() -> None:
    """Orchestrate the full churn pipeline: validate then train."""
    validate_data()
    train_model()


if __name__ == "__main__":
    churn_pipeline()
