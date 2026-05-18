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

import numpy as np
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
        if "close" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'close' column.")

        work_df = df.copy()
        sort_columns = [col for col in ("@timestamp", "date") if col in work_df.columns]
        if sort_columns:
            work_df = work_df.sort_values(sort_columns)

        work_df = self._add_moving_averages(work_df)
        work_df = self._add_ema(work_df)
        work_df = self._add_rsi(work_df)
        work_df = self._add_macd(work_df)
        work_df = self._add_volatility(work_df)
        work_df = self._add_lag_features(work_df)
        work_df = self._add_daily_return(work_df)
        return work_df.dropna().reset_index(drop=True)

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA columns: sma_10, sma_20, sma_50."""
        work_df = df.copy()
        close = work_df["close"].astype(float)
        work_df["sma_10"] = close.rolling(window=10, min_periods=10).mean()
        work_df["sma_20"] = close.rolling(window=20, min_periods=20).mean()
        work_df["sma_50"] = close.rolling(window=50, min_periods=50).mean()
        return work_df

    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ema_20 column."""
        work_df = df.copy()
        close = work_df["close"].astype(float)
        work_df["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
        return work_df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rsi column (14-period)."""
        work_df = df.copy()
        close = work_df["close"].astype(float)
        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)

        avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        gain_only_mask = (avg_loss == 0.0) & (avg_gain > 0.0)
        flat_mask = (avg_loss == 0.0) & (avg_gain == 0.0)
        rsi = rsi.mask(gain_only_mask, 100.0)
        rsi = rsi.mask(flat_mask, 50.0)
        work_df["rsi"] = rsi
        return work_df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add macd, macd_signal, macd_hist columns."""
        work_df = df.copy()
        close = work_df["close"].astype(float)
        ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        work_df["macd"] = macd
        work_df["macd_signal"] = signal
        work_df["macd_hist"] = macd - signal
        return work_df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility column (20-day rolling std of returns)."""
        work_df = df.copy()
        daily_return = work_df["close"].astype(float).pct_change()
        work_df["volatility"] = daily_return.rolling(window=20, min_periods=20).std()
        return work_df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add close_lag_1 and close_lag_5 columns."""
        work_df = df.copy()
        close = work_df["close"].astype(float)
        work_df["close_lag_1"] = close.shift(1)
        work_df["close_lag_5"] = close.shift(5)
        return work_df

    def _add_daily_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add daily_return column (pct change of close)."""
        work_df = df.copy()
        work_df["daily_return"] = work_df["close"].astype(float).pct_change()
        return work_df
