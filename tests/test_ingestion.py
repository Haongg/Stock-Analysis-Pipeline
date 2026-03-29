"""Tests for StockDataFetcher interface."""

import pandas as pd
import pytest

from src.ingestion.data_fetcher import StockDataFetcher


def test_fetch_historical_raises_not_implemented():
    fetcher = StockDataFetcher()
    with pytest.raises(NotImplementedError):
        fetcher.fetch_historical("AAPL", start="2024-01-01")


def test_fetch_latest_raises_not_implemented():
    fetcher = StockDataFetcher()
    with pytest.raises(NotImplementedError):
        fetcher.fetch_latest("AAPL")


def test_save_to_db_raises_not_implemented():
    fetcher = StockDataFetcher()
    with pytest.raises(NotImplementedError):
        fetcher.save_to_db(pd.DataFrame())
