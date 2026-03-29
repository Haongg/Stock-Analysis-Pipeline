"""Tests for the model training module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.models.trainer import FEATURE_COLS, StockModelTrainer


@pytest.fixture()
def sample_features() -> pd.DataFrame:
    """Synthetic feature DataFrame that mimics FeatureEngineer output."""
    rng = np.random.default_rng(0)
    n = 200
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    data = {col: rng.random(n) for col in FEATURE_COLS}
    data["close"] = close
    return pd.DataFrame(data, index=dates)


@patch("src.models.trainer.mlflow")
def test_train_returns_model_and_metrics(mock_mlflow, sample_features):
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "test-run-123"
    mock_mlflow.start_run.return_value = mock_run

    trainer = StockModelTrainer(
        params={"n_estimators": 10, "max_depth": 3, "random_state": 42},
    )
    model, metrics = trainer.train(sample_features)

    assert isinstance(model, RandomForestRegressor)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


@patch("src.models.trainer.mlflow")
def test_train_logs_params_and_metrics(mock_mlflow, sample_features):
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "test-run-456"
    mock_mlflow.start_run.return_value = mock_run

    trainer = StockModelTrainer(
        params={"n_estimators": 5, "max_depth": 2, "random_state": 0},
    )
    trainer.train(sample_features)

    mock_mlflow.log_params.assert_called_once()
    mock_mlflow.log_metrics.assert_called_once()


@patch("src.models.trainer.mlflow")
def test_scaler_fitted_after_train(mock_mlflow, sample_features):
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "test-run-789"
    mock_mlflow.start_run.return_value = mock_run

    trainer = StockModelTrainer(
        params={"n_estimators": 5, "max_depth": 2, "random_state": 0},
    )
    trainer.train(sample_features)
    scaler = trainer.get_scaler()
    # After fitting, scaler should have mean_ attribute
    assert hasattr(scaler, "mean_")
