"""Tests for FeatureEngineer transformations."""

import math
import pandas as pd

from src.features.feature_engineering import FeatureEngineer


def test_transform_adds_indicator_columns_and_drops_nan_rows():
    engineer = FeatureEngineer()
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=80, freq="D"),
            "open": [100.0 + i for i in range(80)],
            "high": [101.0 + i for i in range(80)],
            "low": [99.0 + i for i in range(80)],
            "close": [100.0 + i for i in range(80)],
            "volume": [1_000_000 + i for i in range(80)],
        }
    )

    result = engineer.transform(df)

    expected_columns = {
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
    }
    assert expected_columns.issubset(result.columns)
    assert not result.empty
    assert not result.isna().any().any()


def test_transform_requires_close_column():
    engineer = FeatureEngineer()
    df = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
    try:
        engineer.transform(df)
    except ValueError as exc:
        assert "close" in str(exc)
    else:
        raise AssertionError("Expected ValueError when close column is missing")


def test_transform_macd_hist_matches_difference():
    engineer = FeatureEngineer()
    df = pd.DataFrame(
        {
            "close": [100.0 + i for i in range(80)],
        }
    )
    result = engineer.transform(df)
    last_row = result.iloc[-1]
    assert math.isclose(last_row["macd"] - last_row["macd_signal"], last_row["macd_hist"])
