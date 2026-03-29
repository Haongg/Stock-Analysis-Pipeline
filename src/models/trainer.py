"""
models/trainer.py

Responsibility: train the stock-price prediction model and log everything
to MLflow (parameters, metrics, model artifact, model alias).

Model: Random Forest regressor (next-day close price target).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Feature columns expected in the input DataFrame
FEATURE_COLS = [
    "open", "high", "low", "volume",
    "sma_10", "sma_20", "sma_50", "ema_20",
    "rsi", "macd", "macd_signal", "macd_hist",
    "volatility", "close_lag_1", "close_lag_5", "daily_return",
]
TARGET_COL = "close"

DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}


class StockModelTrainer:
    """Trains and registers a Random Forest model via MLflow."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: str = "stock_analysis",
    ) -> None:
        self.params = params or DEFAULT_PARAMS.copy()
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name

    def train(
        self, df: pd.DataFrame
    ) -> Tuple[RandomForestRegressor, Dict[str, float]]:
        """Train the model on *df*, log to MLflow, return (model, metrics).

        Metrics logged: rmse, mae, r2.
        The registered model version receives the ``champion`` alias.
        """
        raise NotImplementedError

    def _prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Split *df* into feature matrix X and next-day close target y."""
        raise NotImplementedError

    @staticmethod
    def _compute_metrics(y_true, y_pred) -> Dict[str, float]:
        """Return dict with keys ``rmse``, ``mae``, ``r2``."""
        raise NotImplementedError
