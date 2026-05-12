"""Training pipeline for churn prediction model with MLflow tracking.

REPRODUCIBILITY GUARANTEE:
    All sources of randomness are seeded with RANDOM_SEED (=42):
      - sklearn train_test_split (data split)
      - SMOTENC.random_state (synthetic minority data)
      - LightGBM MODEL_PARAMS["random_state"] (tree building)

    Verified: Running training 3 times on identical data produces
    identical metrics down to 6 decimal places (May 12, 2026).

    To re-verify: Run `python -m src.churn_mlops.models.train` 3 times.
    All MLflow runs should show identical accuracy, precision, recall,
    f1, and roc_auc.
"""

from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from imblearn.over_sampling import SMOTENC
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.churn_mlops.data.loader import load_churn_data

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/train_mini.csv")
TARGET_COLUMN = "Churn"
TEST_SIZE = 0.2
RANDOM_SEED = 42

# MLflow configuration
MLFLOW_TRACKING_URI = "file:./mlruns"  # local folder, no server needed
MLFLOW_EXPERIMENT_NAME = "churn-prediction"

# LightGBM hyperparameters
MODEL_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 5,
    "verbose": -1,
    "random_state": RANDOM_SEED,
}

NUM_BOOST_ROUND = 100
EARLY_STOPPING_ROUNDS = 10

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


# ============================================================
# PIPELINE FUNCTIONS
# ============================================================


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target column from feature columns."""
    X = df.drop(columns=[TARGET_COLUMN, "id"])
    y = df[TARGET_COLUMN]
    y_encoded = (y == "Yes").astype(int)

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {y_encoded.value_counts().to_dict()}")
    return X, y_encoded


def prepare_categorical_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns to pandas 'category' dtype for LightGBM."""
    X = X.copy()
    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].astype("category")
    return X


def balance_classes_with_smote(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTENC to balance classes in training data only."""
    logger.info("Applying SMOTENC to balance training data...")
    logger.info(f"   Before SMOTE: {y_train.value_counts().to_dict()}")

    categorical_indices = [X_train.columns.get_loc(col) for col in CATEGORICAL_COLUMNS]

    X_train_for_smote = X_train.copy()
    for col in CATEGORICAL_COLUMNS:
        X_train_for_smote[col] = X_train_for_smote[col].astype("object")

    smote = SMOTENC(
        categorical_features=categorical_indices,
        random_state=RANDOM_SEED,
        k_neighbors=3,
    )
    X_resampled, y_resampled = smote.fit_resample(X_train_for_smote, y_train)
    X_resampled = prepare_categorical_columns(X_resampled)

    logger.success(f"   After SMOTE:  {y_resampled.value_counts().to_dict()}")
    logger.info(f"   New training size: {len(X_resampled)} rows")
    return X_resampled, y_resampled


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> lgb.Booster:
    """Train a LightGBM classifier with early stopping."""
    logger.info("Building LightGBM datasets...")
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_COLUMNS)
    val_dataset = lgb.Dataset(
        X_test,
        label=y_test,
        categorical_feature=CATEGORICAL_COLUMNS,
        reference=train_dataset,
    )

    logger.info(f"Training with {len(X_train)} samples, {X_train.shape[1]} features")

    model = lgb.train(
        params=MODEL_PARAMS,
        train_set=train_dataset,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_dataset, val_dataset],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=0),
        ],
    )

    logger.success(f"✅ Training complete. Best iteration: {model.best_iteration}")
    return model


def evaluate_model(model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate model on test set with multiple metrics."""
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    logger.success("✅ Evaluation complete:")
    for name, value in metrics.items():
        logger.info(f"   {name:>10}: {value:.4f}")
    return metrics


def setup_mlflow() -> None:
    """Configure MLflow tracking — local folder + named experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")


def main() -> None:
    """Run the full training pipeline with MLflow tracking."""
    logger.info("=" * 60)
    logger.info("Starting churn training pipeline")
    logger.info("=" * 60)

    # Step 0: Configure MLflow (NEW)
    setup_mlflow()

    # Wrap entire training in an MLflow run context
    with mlflow.start_run() as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Step 1: Load + validate
        df = load_churn_data(DATA_PATH)

        # Step 2: Separate features + target
        X, y = split_features_and_target(df)

        # Step 3: Convert categoricals
        X = prepare_categorical_columns(X)

        # Step 4: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        logger.info(f"Train: {len(X_train)} rows ({y_train.sum()} churners)")
        logger.info(f"Test:  {len(X_test)} rows ({y_test.sum()} churners)")

        # Step 5: SMOTE balancing
        X_train_balanced, y_train_balanced = balance_classes_with_smote(X_train, y_train)

        # Step 6: Train
        model = train_model(X_train_balanced, y_train_balanced, X_test, y_test)

        # Step 7: Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Step 8: LOG EVERYTHING TO MLFLOW (NEW)
        # 8a. Log all hyperparameters
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("smote_k_neighbors", 3)
        mlflow.log_param("data_path", str(DATA_PATH))
        mlflow.log_param("train_size_after_smote", len(X_train_balanced))
        mlflow.log_param("best_iteration", model.best_iteration)

        # 8b. Log all metrics
        mlflow.log_metrics(metrics)

        # 8c. Log the trained model itself as an artifact
        mlflow.lightgbm.log_model(model, name="model")

        # 8d. Tag the run for easy filtering later
        mlflow.set_tag("model_type", "lightgbm")
        mlflow.set_tag("balancing", "smotenc")
        mlflow.set_tag("dataset", "train_mini")

        logger.success(f"✅ Logged to MLflow: run_id={run.info.run_id}")

    logger.info("=" * 60)
    logger.success("🎉 Pipeline complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
