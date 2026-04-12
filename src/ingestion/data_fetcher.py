"""
ingestion/data_fetcher.py

Responsibility: fetch OHLCV stock data from an external source and
optionally persist it to the database.
"""

from __future__ import annotations # runtime type checking is disabled

import os
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine # connect python to database


class StockDataFetcher:
    """Fetches historical and real-time stock price data."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self._engine: Optional[Engine] = None

    def _get_engine(self) -> Engine:
        """Create and cache a SQLAlchemy engine from DATABASE_URL."""
        if self._engine is not None:
            return self._engine
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL is not configured. "
                "Set DATABASE_URL or pass database_url to StockDataFetcher."
            )
        self._engine = create_engine(self.database_url)
        return self._engine

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Normalize raw yfinance output to open/high/low/close/volume/ticker schema."""
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "ticker"])

        normalized = df.copy()

        # yfinance can return MultiIndex columns depending on params/version.
        if isinstance(normalized.columns, pd.MultiIndex):
            normalized.columns = normalized.columns.get_level_values(0)

        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        normalized = normalized.rename(columns=rename_map)

        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in normalized.columns]
        if missing_cols:
            raise ValueError(f"Missing OHLCV columns from data source: {missing_cols}")

        normalized = normalized[required_cols].copy()
        normalized["ticker"] = ticker.upper()
        normalized.index = pd.to_datetime(normalized.index)
        normalized.index.name = "date"
        return normalized.sort_index()

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
        if not ticker:
            raise ValueError("ticker must be provided")
        if not start:
            raise ValueError("start date must be provided")

        raw = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False, # not auto adjust price
            progress=False, # no progress bar
            group_by="column", # group by OHLCV column
            threads=False, # download data in parallel off
        )
        return self._normalize_ohlcv(raw, ticker=ticker)

    def fetch_latest(self, ticker: str, lookback_days: int = 30) -> pd.DataFrame:
        """Return the most recent *lookback_days* days of data for *ticker*."""
        if lookback_days <= 0:
            raise ValueError("lookback_days must be > 0")
        # Timestamp usually in the mid of the day, so set it to 00:00:00 and plus 1 day to get all data of that day
        end_ts = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
        # Take more data than needed because of weekends and holidays
        start_ts = end_ts - pd.Timedelta(days=max(lookback_days * 2, lookback_days + 7))
        df = self.fetch_historical(
            ticker=ticker,
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
            interval="1d",
        )
        if df.empty:
            return df
        return df.tail(lookback_days)

    def save_to_db(self, df: pd.DataFrame, table_name: str = "stock_prices") -> None:
        """Persist *df* to *table_name* in the configured database."""
        if df.empty:
            return
        if not table_name:
            raise ValueError("table_name must be provided")

        engine = self._get_engine()
        data_to_save = df.copy()
        if not isinstance(data_to_save.index, pd.DatetimeIndex):
            data_to_save.index = pd.to_datetime(data_to_save.index)
        data_to_save.index.name = data_to_save.index.name or "date"

        data_to_save.to_sql(
            table_name,
            con=engine, # connection
            if_exists="append",
            index=True,
            method="multi", # use multiple rows insert
        )
if __name__ == "__main__":
    fetcher = StockDataFetcher()
    # df = fetcher.fetch_historical("AAPL", start="2025-01-1")
    df = fetcher.fetch_latest("AAPL", lookback_days=5)
    print(df.head())