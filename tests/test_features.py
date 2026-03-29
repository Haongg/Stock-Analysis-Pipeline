"""Tests for the feature engineering module."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.features.feature_engineering import FeatureEngineer


@pytest.fixture()
def price_series() -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with 100 rows."""
    rng = np.random.default_rng(42)
    n = 100
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open":   close * (1 + rng.normal(0, 0.005, n)),
            "high":   close * (1 + np.abs(rng.normal(0, 0.01, n))),
            "low":    close * (1 - np.abs(rng.normal(0, 0.01, n))),
            "close":  close,
            "volume": rng.integers(500_000, 2_000_000, n),
        },
        index=dates,
    )


def test_transform_adds_sma_columns(price_series):
    engineer = FeatureEngineer()
    result = engineer.transform(price_series)
    for window in [10, 20, 50]:
        assert f"sma_{window}" in result.columns


def test_transform_adds_ema_column(price_series):
    engineer = FeatureEngineer()
    result = engineer.transform(price_series)
    assert "ema_20" in result.columns


def test_transform_adds_rsi_column(price_series):
    engineer = FeatureEngineer()
    result = engineer.transform(price_series)
    assert "rsi" in result.columns
    assert result["rsi"].between(0, 100).all(), "RSI must be in [0, 100]"


def test_transform_adds_macd_columns(price_series):
    engineer = FeatureEngineer()
    result = engineer.transform(price_series)
    for col in ("macd", "macd_signal", "macd_hist"):
        assert col in result.columns


def test_transform_adds_volatility(price_series):
    engineer = FeatureEngineer()
    result = engineer.transform(price_series)
    assert "volatility" in result.columns
    assert (result["volatility"] >= 0).all(), "Volatility must be non-negative"


def test_transform_adds_lag_features(price_series):
    engineer = FeatureEngineer()
    result = engineer.transform(price_series)
    for lag in [1, 5]:
        assert f"close_lag_{lag}" in result.columns


def test_transform_drops_nan_rows(price_series):
    engineer = FeatureEngineer()
    result = engineer.transform(price_series)
    assert not result.isnull().any().any(), "No NaN values should remain after transform"


def test_transform_empty_df():
    engineer = FeatureEngineer()
    result = engineer.transform(pd.DataFrame())
    assert result.empty
