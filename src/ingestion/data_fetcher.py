"""
ingestion/data_fetcher.py

Responsibility: fetch OHLCV stock data from an external source and
optionally persist it to the database.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


class StockDataFetcher:
    """Fetches historical and real-time stock price data."""

    def fetch_historical(
        self,
        ticker: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return a DataFrame of OHLCV rows for *ticker* in [start, end].

        Columns: open, high, low, close, volume, ticker.
        Index: DatetimeIndex.
        """
        raise NotImplementedError

    def fetch_latest(self, ticker: str, lookback_days: int = 30) -> pd.DataFrame:
        """Return the most recent *lookback_days* days of data for *ticker*."""
        raise NotImplementedError

    def save_to_db(self, df: pd.DataFrame, table_name: str = "stock_prices") -> None:
        """Persist *df* to *table_name* in the configured database."""
        raise NotImplementedError
