"""
Feature engineering module.

Computes technical indicators used as ML features:
  - Simple Moving Averages (SMA 10, 20, 50)
  - Exponential Moving Average (EMA 20)
  - Relative Strength Index (RSI 14)
  - MACD & Signal line
  - Rolling Volatility (20-day)
  - Lag features (close_lag_1, close_lag_5)
  - Daily return

Follows the Open/Closed Principle: new indicators can be added as separate
methods without modifying existing ones.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SMA_WINDOWS = [10, 20, 50]
EMA_WINDOW = 20
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLATILITY_WINDOW = 20
LAG_PERIODS = [1, 5]


class FeatureEngineer:
    """Transforms a raw OHLCV DataFrame into a feature-rich dataset."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transformations to *df*.

        Parameters
        ----------
        df:
            DataFrame with at least a ``close`` column and a
            ``DatetimeIndex``.

        Returns
        -------
        pd.DataFrame
            Original DataFrame augmented with technical indicator columns.
            Rows with NaN values (due to warm-up periods) are dropped.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to FeatureEngineer.transform")
            return df

        df = df.copy()
        df = self._add_moving_averages(df)
        df = self._add_ema(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_volatility(df)
        df = self._add_lag_features(df)
        df = self._add_daily_return(df)

        before = len(df)
        df = df.dropna()
        logger.info(
            "Feature engineering complete: %d rows (dropped %d NaN rows)",
            len(df),
            before - len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Individual indicator methods
    # ------------------------------------------------------------------

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in SMA_WINDOWS:
            df[f"sma_{window}"] = df["close"].rolling(window=window).mean()
        return df

    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        df[f"ema_{EMA_WINDOW}"] = (
            df["close"].ewm(span=EMA_WINDOW, adjust=False).mean()
        )
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
        avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()

        rs = avg_gain / avg_loss.replace(0, float("nan"))
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = (
            df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
        )
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        df["volatility"] = (
            df["close"].pct_change().rolling(window=VOLATILITY_WINDOW).std()
        )
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for lag in LAG_PERIODS:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
        return df

    def _add_daily_return(self, df: pd.DataFrame) -> pd.DataFrame:
        df["daily_return"] = df["close"].pct_change()
        return df
