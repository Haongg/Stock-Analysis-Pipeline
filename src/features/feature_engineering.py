"""
features/feature_engineering.py

Responsibility: transform raw OHLCV data into ML-ready features.

Indicators computed:
  - SMA 10 / 20 / 50
  - EMA 20
  - RSI 14
  - MACD + Signal + Histogram
  - Rolling Volatility (20-day)
  - Lag features: close_lag_1, close_lag_5
  - Daily return
"""

from __future__ import annotations

import pandas as pd


class FeatureEngineer:
    """Adds technical indicator columns to a raw OHLCV DataFrame."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return *df* augmented with all indicator columns (NaN rows dropped).

        Parameters
        ----------
        df:
            Raw OHLCV DataFrame with at least a ``close`` column.

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame with no NaN rows.
        """
        raise NotImplementedError

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA columns: sma_10, sma_20, sma_50."""
        raise NotImplementedError

    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ema_20 column."""
        raise NotImplementedError

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rsi column (14-period)."""
        raise NotImplementedError

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add macd, macd_signal, macd_hist columns."""
        raise NotImplementedError

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility column (20-day rolling std of returns)."""
        raise NotImplementedError

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add close_lag_1 and close_lag_5 columns."""
        raise NotImplementedError

    def _add_daily_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add daily_return column (pct change of close)."""
        raise NotImplementedError
