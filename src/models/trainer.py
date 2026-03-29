"""
Model training module.

Trains a Random Forest regressor to predict the next-day closing price.
All hyperparameters and metrics are logged to MLflow for experiment tracking
and model versioning.

Follows Dependency Inversion Principle: the Trainer depends on abstract
interfaces (DataFrames, config dicts) rather than concrete data sources.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}

FEATURE_COLS = [
    "open",
    "high",
    "low",
    "volume",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "volatility",
    "close_lag_1",
    "close_lag_5",
    "daily_return",
]
TARGET_COL = "close"


class StockModelTrainer:
    """Encapsulates the training loop for the stock-price prediction model."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: str = "stock_analysis",
    ) -> None:
        self._params = params or DEFAULT_PARAMS.copy()
        self._experiment_name = experiment_name
        self._scaler = StandardScaler()

        uri = mlflow_tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://mlflow:5000"
        )
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        logger.info("MLflow tracking URI: %s  experiment: %s", uri, experiment_name)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> Tuple[RandomForestRegressor, Dict[str, float]]:
        """Train the model on *df* and log everything to MLflow.

        Parameters
        ----------
        df:
            Feature-engineered DataFrame produced by
            :class:`~src.features.feature_engineering.FeatureEngineer`.

        Returns
        -------
        Tuple[RandomForestRegressor, Dict[str, float]]
            Fitted model and a metrics dictionary
            (``rmse``, ``mae``, ``r2``).
        """
        X, y = self._prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        with mlflow.start_run() as run:
            logger.info("MLflow run id: %s", run.info.run_id)

            # Log hyper-parameters
            mlflow.log_params(self._params)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_param("features", FEATURE_COLS)

            model = RandomForestRegressor(**self._params)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            metrics = self._compute_metrics(y_test, y_pred)

            mlflow.log_metrics(metrics)

            # Log feature importance as a metric per feature
            for feat, imp in zip(FEATURE_COLS, model.feature_importances_):
                mlflow.log_metric(f"importance_{feat}", float(imp))

            mlflow.sklearn.log_model(
                model,
                artifact_path="random_forest",
                registered_model_name="stock_price_predictor",
            )

            logger.info(
                "Training complete – RMSE: %.4f  MAE: %.4f  R²: %.4f",
                metrics["rmse"],
                metrics["mae"],
                metrics["r2"],
            )

        return model, metrics

    def get_scaler(self) -> StandardScaler:
        """Return the fitted scaler (available after :meth:`train` is called)."""
        return self._scaler

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        available = [c for c in FEATURE_COLS if c in df.columns]
        missing = set(FEATURE_COLS) - set(available)
        if missing:
            logger.warning("Missing feature columns: %s", missing)

        X = df[available].copy()
        # Target: next-day close price
        y = df[TARGET_COL].shift(-1).dropna()
        X = X.iloc[: len(y)]
        return X, y

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "r2": r2}
