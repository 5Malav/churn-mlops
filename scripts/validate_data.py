"""Standalone data validation stage for the DVC pipeline.

Loads the raw training data and runs it through the Pandera schema.
Exits 0 if valid, non-zero if validation fails — so DVC knows
whether the validate stage passed before training proceeds.
"""

from pathlib import Path

from loguru import logger

from src.churn_mlops.data.loader import load_churn_data

DATA_PATH = Path("data/raw/train.csv")


def main() -> None:
    """Validate the raw dataset. Raises if schema validation fails."""
    logger.info("=" * 60)
    logger.info("DVC STAGE: validate")
    logger.info("=" * 60)

    df = load_churn_data(DATA_PATH)
    logger.success(f"✅ Validation stage passed: {len(df)} rows valid")


if __name__ == "__main__":
    main()
