"""Tests for the data ingestion module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.data_fetcher import StockDataFetcher


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Minimal OHLCV DataFrame mimicking yfinance output."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [150.0, 151.0, 152.0, 153.0, 154.0],
            "High": [155.0, 156.0, 157.0, 158.0, 159.0],
            "Low": [149.0, 150.0, 151.0, 152.0, 153.0],
            "Close": [152.0, 153.0, 154.0, 155.0, 156.0],
            "Volume": [1_000_000] * 5,
        },
        index=dates,
    )


def test_fetch_historical_returns_dataframe(sample_ohlcv):
    with patch("src.ingestion.data_fetcher.yf.download", return_value=sample_ohlcv):
        fetcher = StockDataFetcher()
        df = fetcher.fetch_historical("AAPL", start="2024-01-01", end="2024-01-06")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "close" in df.columns
    assert df["ticker"].iloc[0] == "AAPL"


def test_fetch_historical_normalizes_columns(sample_ohlcv):
    with patch("src.ingestion.data_fetcher.yf.download", return_value=sample_ohlcv):
        fetcher = StockDataFetcher()
        df = fetcher.fetch_historical("MSFT", start="2024-01-01")

    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns, f"Missing column: {col}"


def test_fetch_historical_empty_response():
    with patch(
        "src.ingestion.data_fetcher.yf.download", return_value=pd.DataFrame()
    ):
        fetcher = StockDataFetcher()
        df = fetcher.fetch_historical("INVALID", start="2024-01-01")

    assert df.empty


def test_fetch_latest_uses_lookback(sample_ohlcv):
    with patch("src.ingestion.data_fetcher.yf.download", return_value=sample_ohlcv) as mock_dl:
        fetcher = StockDataFetcher()
        fetcher.fetch_latest("AAPL", lookback_days=10)

    assert mock_dl.called


def test_save_to_db_raises_without_engine(sample_ohlcv):
    fetcher = StockDataFetcher(engine=None)
    with pytest.raises(RuntimeError, match="No database engine"):
        fetcher.save_to_db(sample_ohlcv)
