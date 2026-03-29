"""
Data ingestion module for fetching historical and real-time stock data.

Follows Single Responsibility Principle (SRP): this module is solely responsible
for data acquisition from external sources.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetches stock price data from Yahoo Finance and persists it to the database."""

    def __init__(self, engine: Optional[Engine] = None) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        ticker: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download historical OHLCV data for *ticker*.

        Parameters
        ----------
        ticker:
            Stock ticker symbol, e.g. ``"AAPL"``.
        start:
            Start date as ``"YYYY-MM-DD"`` string.
        end:
            End date as ``"YYYY-MM-DD"`` string (defaults to today).
        interval:
            Data granularity accepted by yfinance (e.g. ``"1d"``, ``"1h"``).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``open, high, low, close, volume`` and a
            ``DatetimeIndex``.
        """
        end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info("Fetching historical data for %s (%s → %s)", ticker, start, end)

        raw: pd.DataFrame = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            logger.warning("No data returned for ticker %s", ticker)
            return pd.DataFrame()

        df = self._normalize_columns(raw)
        df["ticker"] = ticker.upper()
        logger.info("Fetched %d rows for %s", len(df), ticker)
        return df

    def fetch_latest(self, ticker: str, lookback_days: int = 30) -> pd.DataFrame:
        """Convenience wrapper that fetches recent data up to today.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.
        lookback_days:
            Number of calendar days to look back.
        """
        start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )
        return self.fetch_historical(ticker, start=start)

    def save_to_db(self, df: pd.DataFrame, table_name: str = "stock_prices") -> None:
        """Persist *df* to the configured database table.

        Parameters
        ----------
        df:
            DataFrame produced by :meth:`fetch_historical` or
            :meth:`fetch_latest`.
        table_name:
            Target table name.  The table is created if it does not exist.

        Raises
        ------
        RuntimeError
            If no SQLAlchemy engine was provided at construction time.
        """
        if self._engine is None:
            raise RuntimeError(
                "No database engine configured.  "
                "Pass a SQLAlchemy Engine to StockDataFetcher.__init__."
            )

        df_to_save = df.reset_index()
        df_to_save.columns = [c.lower().replace(" ", "_") for c in df_to_save.columns]

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id         SERIAL PRIMARY KEY,
                        date       TIMESTAMPTZ NOT NULL,
                        ticker     TEXT        NOT NULL,
                        open       DOUBLE PRECISION,
                        high       DOUBLE PRECISION,
                        low        DOUBLE PRECISION,
                        close      DOUBLE PRECISION,
                        volume     BIGINT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
            )

        df_to_save.to_sql(
            table_name,
            self._engine,
            if_exists="append",
            index=False,
            method="multi",
        )
        logger.info("Saved %d rows to table '%s'", len(df_to_save), table_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Flatten multi-level columns (yfinance ≥ 0.2) and lower-case names."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df
