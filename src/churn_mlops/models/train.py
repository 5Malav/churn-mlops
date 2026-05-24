"""Training pipeline for churn prediction model with MLflow tracking.

REPRODUCIBILITY GUARANTEE:
    All sources of randomness are seeded with RANDOM_SEED (=42):
      - sklearn train_test_split (data split)
      - SMOTENC.random_state (synthetic minority data)
      - LightGBM MODEL_PARAMS["random_state"] (tree building)

    Verified: Running training 3 times on identical data produces
    identical metrics down to 6 decimal places (May 12, 2026).
"""

from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import optuna
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

DATA_PATH = Path("data/raw/train.csv")
TARGET_COLUMN = "Churn"
TEST_SIZE = 0.2
RANDOM_SEED = 42


# Subsampling for fast iteration (Phase 2)
# Set to None for full dataset, or an int (e.g., 50000) for stratified subset
SUBSAMPLE_SIZE: int | None = None

# Balancing strategy (Phase 2 — comparison experiments)
# Options: "smote", "class_weight", "none"
BALANCING_STRATEGY: str = "class_weight"

# MLflow configuration
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "churn-prediction"

# Model export (Phase 3 — DVC-tracked production artifact)
MODEL_OUTPUT_PATH = Path("models/churn_model.txt")
# MODEL_PARAMS — tuned via Optuna study (May 18, 2026)
# Best trial: #21 on 50K subsample, F1=0.6984
# Original (Phase 1.8) values kept as comments for reference.
MODEL_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.0101,  # was 0.05 — Optuna chose much lower
    "num_leaves": 169,  # was 15 — Optuna chose much higher
    "min_child_samples": 63,  # was 5 — Optuna chose much higher
    "feature_fraction": 0.5665,  # NEW from Optuna
    "bagging_fraction": 0.6944,  # NEW from Optuna
    "verbose": -1,
    "random_state": RANDOM_SEED,
}

NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 10

# Optuna configuration (Phase 2 — hyperparameter tuning)
# Set RUN_OPTUNA = True to launch tuning study instead of single training
RUN_OPTUNA: bool = False
OPTUNA_N_TRIALS = 30
OPTUNA_OPTIMIZE_METRIC = "f1"  # which metric Optuna maximizes

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


def stratified_subsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Take a stratified subsample preserving class ratios.

    Why: Full 594K dataset takes ~90 min per training run with SMOTE.
    For fast experimentation (Optuna tuning), we subsample to ~50K rows
    while preserving the 30% churn ratio. Industry-standard pattern.
    """
    logger.info(f"Subsampling from {len(df)} rows to ~{n} rows (stratified)")
    fraction = n / len(df)

    subset = df.groupby(TARGET_COLUMN, group_keys=False).apply(
        lambda group: group.sample(
            n=int(len(group) * fraction),
            random_state=RANDOM_SEED,
        )
    )

    logger.info(f"Subsampled size: {len(subset)} rows")
    logger.info(f"Subsample target distribution: {subset[TARGET_COLUMN].value_counts().to_dict()}")
    return subset


def balance_with_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Strategy: Apply SMOTENC to balance training data only."""
    logger.info("Strategy: SMOTE — generating synthetic minority samples")
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


def balance_with_class_weight_unchanged(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Strategy: class_weight — return data unchanged.

    Balancing happens via MODEL_PARAMS["scale_pos_weight"] (added in main()).
    Faster than SMOTE because we skip synthetic data generation.
    """
    logger.info("Strategy: class_weight — model will weight minority class")
    logger.info(f"   Training distribution: {y_train.value_counts().to_dict()}")
    return X_train, y_train


def balance_classes(
    X_train: pd.DataFrame, y_train: pd.Series, strategy: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Dispatcher: route to the right balancing strategy."""
    if strategy == "smote":
        return balance_with_smote(X_train, y_train)
    elif strategy == "class_weight":
        return balance_with_class_weight_unchanged(X_train, y_train)
    elif strategy == "none":
        logger.info("Strategy: none — no balancing applied")
        return X_train, y_train
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'smote', 'class_weight', or 'none'.")


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


def optuna_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scale_pos_weight: float,
) -> float:
    """One Optuna trial: sample params, train, return F1.

    Optuna calls this function once per trial. Each call:
    1. Samples hyperparameters from the search space
    2. Trains a LightGBM model with those params
    3. Evaluates on the test set
    4. Logs to MLflow as a nested run
    5. Returns the metric Optuna will optimize (F1)

    Args:
        trial: Optuna trial object — used to suggest hyperparameters.
        X_train, y_train: Training data.
        X_test, y_test: Validation data (used for evaluation AND early stopping).
        scale_pos_weight: For class imbalance (calculated in main()).

    Returns:
        F1 score on test set — Optuna will maximize this.
    """
    # 1. Sample hyperparameters from the search space
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
        "random_state": RANDOM_SEED,
        "scale_pos_weight": scale_pos_weight,
        # The 5 params we tune
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
    }

    # 2. Log each trial as a nested MLflow run for full visibility
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(params)

        # 3. Train LightGBM with these specific params
        train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_COLUMNS)
        val_dataset = lgb.Dataset(
            X_test,
            label=y_test,
            categorical_feature=CATEGORICAL_COLUMNS,
            reference=train_dataset,
        )

        model = lgb.train(
            params=params,
            train_set=train_dataset,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_dataset],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # 4. Evaluate
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_metric("best_iteration", model.best_iteration)

        # 5. Return the metric Optuna will maximize
        target_metric = metrics[OPTUNA_OPTIMIZE_METRIC]
        logger.info(
            f"Trial {trial.number}: {OPTUNA_OPTIMIZE_METRIC}={target_metric:.4f} | "
            f"lr={params['learning_rate']:.4f}, leaves={params['num_leaves']}, "
            f"min_child={params['min_child_samples']}"
        )
        return target_metric


def run_optuna_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> optuna.Study:
    """Run an Optuna hyperparameter optimization study.

    Returns the completed study object with best_trial accessible.
    """
    # Compute scale_pos_weight (passed to every trial)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight for all trials: {scale_pos_weight:.4f}")

    # Create study — direction="maximize" because we want HIGH F1
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),  # Bayesian, reproducible
        study_name="churn-lgbm-tuning",
    )

    logger.info(f"Starting Optuna study with {OPTUNA_N_TRIALS} trials...")
    logger.info(f"Optimizing for: {OPTUNA_OPTIMIZE_METRIC}")

    # Run the trials
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_test, y_test, scale_pos_weight),
        n_trials=OPTUNA_N_TRIALS,
        show_progress_bar=False,  # we use logger instead
    )

    logger.success("✅ Optuna study complete!")
    logger.success(f"   Best trial: #{study.best_trial.number}")
    logger.success(f"   Best {OPTUNA_OPTIMIZE_METRIC}: {study.best_value:.4f}")
    logger.success(f"   Best params: {study.best_params}")

    return study


def main() -> None:
    """Run the full training pipeline with MLflow tracking."""
    logger.info("=" * 60)
    logger.info("Starting churn training pipeline")
    logger.info("=" * 60)

    setup_mlflow()

    with mlflow.start_run() as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Step 1: Load + validate
        df = load_churn_data(DATA_PATH)

        # Step 1.5: Subsample if configured
        if SUBSAMPLE_SIZE is not None:
            df = stratified_subsample(df, SUBSAMPLE_SIZE)

        # Step 2: Separate features + target
        X, y = split_features_and_target(df)

        # Step 3: Convert categoricals
        X = prepare_categorical_columns(X)

        # Step 4: Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        logger.info(f"Train: {len(X_train)} rows ({y_train.sum()} churners)")
        logger.info(f"Test:  {len(X_test)} rows ({y_test.sum()} churners)")

        # Step 5: Balance classes per configured strategy
        X_train_balanced, y_train_balanced = balance_classes(
            X_train, y_train, strategy=BALANCING_STRATEGY
        )

        # If using class_weight strategy, add scale_pos_weight to MODEL_PARAMS
        if BALANCING_STRATEGY == "class_weight":
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            MODEL_PARAMS["scale_pos_weight"] = scale_pos_weight
            logger.info(f"   scale_pos_weight set to: {scale_pos_weight:.4f}")
        else:
            MODEL_PARAMS.pop("scale_pos_weight", None)

        # Step 6: Either run Optuna tuning OR single training
        if RUN_OPTUNA:
            study = run_optuna_study(X_train_balanced, y_train_balanced, X_test, y_test)
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric(f"best_{OPTUNA_OPTIMIZE_METRIC}", study.best_value)
            mlflow.set_tag("mode", "optuna_tuning")
            mlflow.set_tag("n_trials", str(OPTUNA_N_TRIALS))
            logger.info("=" * 60)
            logger.success("🎉 Optuna study complete")
            logger.info("=" * 60)
            return  # exit early — no single training in tuning mode

        # Single training (existing behavior)
        model = train_model(X_train_balanced, y_train_balanced, X_test, y_test)

        # Step 7: Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Step 7.5: Export model as DVC-tracked production artifact (Phase 3)
        MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_OUTPUT_PATH))
        logger.success(f"✅ Model saved to {MODEL_OUTPUT_PATH}")

        # Step 8: Log everything to MLflow
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("subsample_size", SUBSAMPLE_SIZE)
        mlflow.log_param("balancing_strategy", BALANCING_STRATEGY)
        mlflow.log_param("data_path", str(DATA_PATH))
        mlflow.log_param("train_size_after_balancing", len(X_train_balanced))
        mlflow.log_param("best_iteration", model.best_iteration)

        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, name="model")

        mlflow.set_tag("model_type", "lightgbm")
        mlflow.set_tag("balancing", BALANCING_STRATEGY)
        mlflow.set_tag(
            "dataset",
            "train_full_subsampled" if SUBSAMPLE_SIZE else "train_full",
        )

        logger.success(f"✅ Logged to MLflow: run_id={run.info.run_id}")

    logger.info("=" * 60)
    logger.success("🎉 Pipeline complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
