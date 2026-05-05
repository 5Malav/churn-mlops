"""Data loading + validation entry point for churn dataset."""

from pathlib import Path

import pandas as pd
import pandera.errors as pa_errors
from loguru import logger

from src.churn_mlops.data.schema import ChurnSchema


def load_churn_data(csv_path: str | Path) -> pd.DataFrame:
    """Load a churn CSV and validate it against ChurnSchema.

    This is the ONLY approved way to load training data into the pipeline.
    Never use pd.read_csv() directly — it bypasses validation.

    Args:
        csv_path: Path to the CSV file (string or Path object).

    Returns:
        Validated pandas DataFrame matching ChurnSchema.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        pandera.errors.SchemaError: If data violates the schema.
    """
    csv_path = Path(csv_path)

    # Step 1: Verify file exists (fail fast with clear message)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Loading data from {csv_path}")

    # Step 2: Load the CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")

    # Step 3: Validate against schema (this is the bouncer)
    try:
        validated_df = ChurnSchema.validate(df, lazy=True)
        logger.success(f"✅ Validation passed: {len(validated_df)} rows")
        return validated_df
    except pa_errors.SchemaErrors as exc:
        # lazy=True collects ALL errors; we surface them clearly
        logger.error(f"❌ Validation failed with {len(exc.failure_cases)} issues")
        logger.error(f"\n{exc.failure_cases}")
        raise
