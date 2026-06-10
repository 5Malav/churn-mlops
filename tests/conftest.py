"""Shared pytest setup.

Creates a tiny dummy model so API tests can run anywhere — including CI,
where the real DVC-tracked model file isn't present.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

MODEL_PATH = Path("models/churn_model.txt")


@pytest.fixture(autouse=True, scope="session")
def ensure_model_exists():
    """If the real model isn't present (e.g. on CI), train a tiny stand-in."""
    if MODEL_PATH.exists():
        return  # real model is here (local dev) — use it

    # No model (CI): build a minimal throwaway one with the right shape.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 19 features, matching the API's CustomerFeatures schema.
    rng = np.random.default_rng(42)
    X = rng.random((100, 19))
    y = rng.integers(0, 2, size=100)

    train_data = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "verbose": -1}
    model = lgb.train(params, train_data, num_boost_round=5)
    model.save_model(str(MODEL_PATH))
