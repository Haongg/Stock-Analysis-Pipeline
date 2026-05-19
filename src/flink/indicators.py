from __future__ import annotations

import math
from typing import Sequence


class IndicatorCalculator:
    """Pure-Python technical indicator utilities for Flink processing.

    Input series must be ordered from oldest -> newest.
    For insufficient data, functions return ``float("nan")`` (or tuple of NaN).
    Invalid configuration (e.g., non-positive period) raises ``ValueError``.
    """

    @staticmethod
    def _validate_period(period: int) -> None:
        if period <= 0:
            raise ValueError("period must be > 0")

    @staticmethod
    def _to_float_list(prices: Sequence[float]) -> list[float]:
        return [float(p) for p in prices]

    @staticmethod
    def _has_enough(values: Sequence[float], required: int) -> bool:
        return len(values) >= required

    @staticmethod
    def calculate_sma(prices: Sequence[float], period: int) -> float:
        """Simple moving average over the latest ``period`` prices."""
        IndicatorCalculator._validate_period(period)
        values = IndicatorCalculator._to_float_list(prices)
        if not IndicatorCalculator._has_enough(values, period):
            return float("nan")
        window = values[-period:]
        return sum(window) / float(period)

    @staticmethod
    def calculate_sma_bundle(prices: Sequence[float]) -> dict[str, float]:
        """Return standard SMA bundle used downstream in feature enrichment.

        Output keys:
        - ``sma_10``
        - ``sma_20``
        - ``sma_50``
        """
        return {
            "sma_10": IndicatorCalculator.calculate_sma(prices, 10),
            "sma_20": IndicatorCalculator.calculate_sma(prices, 20),
            "sma_50": IndicatorCalculator.calculate_sma(prices, 50),
        }

    @staticmethod
    def calculate_ema(prices: Sequence[float], period: int) -> float:
        """Exponential moving average.

        Seed EMA with SMA of the first ``period`` values, then apply:
        EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}
        where alpha = 2 / (period + 1).
        """
        IndicatorCalculator._validate_period(period)
        values = IndicatorCalculator._to_float_list(prices)
        if not IndicatorCalculator._has_enough(values, period):
            return float("nan")

        alpha = 2.0 / float(period + 1)
        ema = sum(values[:period]) / float(period)
        for price in values[period:]:
            ema = alpha * price + (1.0 - alpha) * ema
        return ema

    @staticmethod
    def calculate_rsi(prices: Sequence[float], period: int = 14) -> float:
        """Relative Strength Index (Wilder smoothing)."""
        IndicatorCalculator._validate_period(period)
        values = IndicatorCalculator._to_float_list(prices)
        if not IndicatorCalculator._has_enough(values, period + 1):
            return float("nan")

        deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
        gains = [max(delta, 0.0) for delta in deltas]
        losses = [max(-delta, 0.0) for delta in deltas]

        avg_gain = sum(gains[:period]) / float(period)
        avg_loss = sum(losses[:period]) / float(period)

        for i in range(period, len(deltas)):
            avg_gain = ((avg_gain * (period - 1)) + gains[i]) / float(period)
            avg_loss = ((avg_loss * (period - 1)) + losses[i]) / float(period)

        if avg_loss == 0.0:
            if avg_gain > 0.0:
                return 100.0
            return 50.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def calculate_macd(prices: Sequence[float]) -> tuple[float, float, float]:
        """MACD (12,26,9) -> (macd, signal, histogram)."""
        values = IndicatorCalculator._to_float_list(prices)
        # Need enough points for EMA26 and stable signal EMA9 from MACD series.
        if len(values) < 35:
            nan = float("nan")
            return nan, nan, nan

        macd_series: list[float] = []
        for i in range(26, len(values) + 1):
            sub = values[:i]
            ema12 = IndicatorCalculator.calculate_ema(sub, 12)
            ema26 = IndicatorCalculator.calculate_ema(sub, 26)
            if math.isnan(ema12) or math.isnan(ema26):
                continue
            macd_series.append(ema12 - ema26)

        if len(macd_series) < 9:
            nan = float("nan")
            return nan, nan, nan

        macd_value = macd_series[-1]
        signal_value = IndicatorCalculator.calculate_ema(macd_series, 9)
        if math.isnan(signal_value):
            nan = float("nan")
            return nan, nan, nan
        hist_value = macd_value - signal_value
        return macd_value, signal_value, hist_value

    @staticmethod
    def calculate_volatility(prices: Sequence[float], period: int = 20) -> float:
        """Rolling volatility = sample std (ddof=1) of latest ``period`` returns."""
        IndicatorCalculator._validate_period(period)
        values = IndicatorCalculator._to_float_list(prices)
        if not IndicatorCalculator._has_enough(values, period + 1):
            return float("nan")

        returns: list[float] = []
        for i in range(1, len(values)):
            prev = values[i - 1]
            curr = values[i]
            if prev == 0.0:
                return float("nan")
            returns.append((curr - prev) / prev)

        if len(returns) < period:
            return float("nan")

        window = returns[-period:]
        mean_ret = sum(window) / float(period)
        var = sum((r - mean_ret) ** 2 for r in window) / float(period - 1)
        return math.sqrt(var)

    @staticmethod
    def calculate_daily_return(prices: Sequence[float]) -> float:
        """Daily return using the latest close and previous close."""
        values = IndicatorCalculator._to_float_list(prices)
        if not IndicatorCalculator._has_enough(values, 2):
            return float("nan")

        previous_close = values[-2]
        if previous_close == 0.0:
            return float("nan")
        return (values[-1] - previous_close) / previous_close

    @staticmethod
    def calculate_lag_features(prices: Sequence[float]) -> dict[str, float]:
        """Return close lag features and daily return for the latest close."""
        values = IndicatorCalculator._to_float_list(prices)
        return {
            "close_lag_1": values[-2] if IndicatorCalculator._has_enough(values, 2) else float("nan"),
            "close_lag_5": values[-6] if IndicatorCalculator._has_enough(values, 6) else float("nan"),
            "daily_return": IndicatorCalculator.calculate_daily_return(values),
        }
