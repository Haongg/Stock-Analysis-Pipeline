import math

from src.flink.indicators import IndicatorCalculator


def test_sma_10_20_50_with_sufficient_data() -> None:
    prices = list(range(1, 61))  # 1..60
    assert IndicatorCalculator.calculate_sma(prices, 10) == sum(range(51, 61)) / 10.0
    assert IndicatorCalculator.calculate_sma(prices, 20) == sum(range(41, 61)) / 20.0
    assert IndicatorCalculator.calculate_sma(prices, 50) == sum(range(11, 61)) / 50.0


def test_sma_returns_nan_for_10_20_50_when_insufficient_data() -> None:
    prices = list(range(1, 10))
    assert math.isnan(IndicatorCalculator.calculate_sma(prices, 10))
    assert math.isnan(IndicatorCalculator.calculate_sma(prices, 20))
    assert math.isnan(IndicatorCalculator.calculate_sma(prices, 50))


def test_sma_uses_latest_period_window() -> None:
    prices = [1.0] * 90 + [100.0] * 10
    assert IndicatorCalculator.calculate_sma(prices, 10) == 100.0


def test_calculate_ema_basic() -> None:
    prices = [1, 2, 3, 4, 5]
    ema = IndicatorCalculator.calculate_ema(prices, 3)
    # Seed SMA(1,2,3)=2; then ema4=3; ema5=4
    assert abs(ema - 4.0) < 1e-12


def test_calculate_ema_uses_recursive_weighting() -> None:
    prices = [10.0, 11.0, 12.0, 13.0, 20.0]
    ema = IndicatorCalculator.calculate_ema(prices, 3)
    # Seed at 11, then 12, then 16
    assert abs(ema - 16.0) < 1e-12


def test_calculate_rsi_gain_only_and_flat() -> None:
    gain_only = list(range(1, 30))
    rsi_gain = IndicatorCalculator.calculate_rsi(gain_only, 14)
    assert rsi_gain == 100.0

    flat = [100.0] * 30
    rsi_flat = IndicatorCalculator.calculate_rsi(flat, 14)
    assert rsi_flat == 50.0


def test_calculate_rsi_stays_within_expected_range() -> None:
    prices = [44, 44.15, 43.9, 44.35, 44.8, 44.1, 43.75, 44.5, 45.1, 44.95, 45.4, 45.0, 45.6, 45.2, 45.75, 46.1, 45.8, 46.25]
    rsi = IndicatorCalculator.calculate_rsi(prices, 14)
    assert not math.isnan(rsi)
    assert 0.0 <= rsi <= 100.0


def test_calculate_macd_shape_and_nan_on_short_series() -> None:
    short = [100.0 + i for i in range(20)]
    macd, signal, hist = IndicatorCalculator.calculate_macd(short)
    assert math.isnan(macd)
    assert math.isnan(signal)
    assert math.isnan(hist)

    long_series = [100.0 + i for i in range(60)]
    macd, signal, hist = IndicatorCalculator.calculate_macd(long_series)
    assert not math.isnan(macd)
    assert not math.isnan(signal)
    assert not math.isnan(hist)
    assert abs((macd - signal) - hist) < 1e-10


def test_calculate_macd_trending_series_positive_histogram() -> None:
    prices = [100.0 + (i * 0.8) for i in range(80)]
    macd, signal, hist = IndicatorCalculator.calculate_macd(prices)
    assert macd > 0.0
    assert signal > 0.0
    assert hist >= 0.0


def test_calculate_volatility_basic() -> None:
    prices = [100, 101, 100, 102, 101, 103, 104, 102, 103, 105, 104, 106, 107, 106, 108, 109, 108, 110, 111, 112, 113]
    vol = IndicatorCalculator.calculate_volatility(prices, 20)
    assert not math.isnan(vol)
    assert vol > 0.0


def test_calculate_volatility_matches_sample_std_of_returns() -> None:
    prices = [100.0, 102.0, 101.0, 104.0, 108.0, 107.0]
    returns = [
        (prices[i] - prices[i - 1]) / prices[i - 1]
        for i in range(1, len(prices))
    ]
    mean_return = sum(returns) / len(returns)
    expected = math.sqrt(sum((ret - mean_return) ** 2 for ret in returns) / (len(returns) - 1))

    assert abs(IndicatorCalculator.calculate_volatility(prices, period=5) - expected) < 1e-12


def test_calculate_volatility_uses_latest_return_window() -> None:
    prices = [100.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0]
    latest_returns = [
        (prices[i] - prices[i - 1]) / prices[i - 1]
        for i in range(2, len(prices))
    ]
    mean_return = sum(latest_returns) / len(latest_returns)
    expected = math.sqrt(
        sum((ret - mean_return) ** 2 for ret in latest_returns) / (len(latest_returns) - 1)
    )

    assert abs(IndicatorCalculator.calculate_volatility(prices, period=5) - expected) < 1e-12


def test_calculate_volatility_returns_nan_for_zero_previous_close() -> None:
    prices = [100.0, 101.0, 0.0, 102.0, 103.0, 104.0]

    assert math.isnan(IndicatorCalculator.calculate_volatility(prices, period=5))


def test_calculate_daily_return_exact_value() -> None:
    prices = [100.0, 110.0]

    assert IndicatorCalculator.calculate_daily_return(prices) == 0.1


def test_calculate_daily_return_returns_nan_for_short_or_zero_previous_close() -> None:
    assert math.isnan(IndicatorCalculator.calculate_daily_return([100.0]))
    assert math.isnan(IndicatorCalculator.calculate_daily_return([0.0, 100.0]))


def test_calculate_lag_features_returns_expected_values() -> None:
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 110.0]
    features = IndicatorCalculator.calculate_lag_features(prices)

    assert features["close_lag_1"] == 105.0
    assert features["close_lag_5"] == 101.0
    assert abs(features["daily_return"] - ((110.0 - 105.0) / 105.0)) < 1e-12


def test_calculate_lag_features_returns_nan_for_missing_lag_windows() -> None:
    features = IndicatorCalculator.calculate_lag_features([100.0, 101.0])

    assert features["close_lag_1"] == 100.0
    assert math.isnan(features["close_lag_5"])
    assert features["daily_return"] == 0.01


def test_calculate_lag_features_coerces_int_and_float_inputs() -> None:
    features = IndicatorCalculator.calculate_lag_features([100, 101.5, 103, 104.5, 106, 107.5])

    assert isinstance(features["close_lag_1"], float)
    assert features["close_lag_1"] == 106.0
    assert features["close_lag_5"] == 100.0


def test_insufficient_data_returns_nan() -> None:
    prices = [1.0, 2.0]
    assert math.isnan(IndicatorCalculator.calculate_sma(prices, 5))
    assert math.isnan(IndicatorCalculator.calculate_ema(prices, 5))
    assert math.isnan(IndicatorCalculator.calculate_rsi(prices, 14))
    assert math.isnan(IndicatorCalculator.calculate_volatility(prices, 20))
    assert math.isnan(IndicatorCalculator.calculate_daily_return([1.0]))


def test_sma_invalid_period_raises_value_error() -> None:
    prices = [1.0, 2.0, 3.0]
    try:
        IndicatorCalculator.calculate_sma(prices, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for period=0 in SMA")

    try:
        IndicatorCalculator.calculate_sma(prices, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for period=-1 in SMA")


def test_sma_bundle_returns_expected_keys_and_values() -> None:
    prices = list(range(1, 61))
    bundle = IndicatorCalculator.calculate_sma_bundle(prices)
    assert set(bundle.keys()) == {"sma_10", "sma_20", "sma_50"}
    assert bundle["sma_10"] == IndicatorCalculator.calculate_sma(prices, 10)
    assert bundle["sma_20"] == IndicatorCalculator.calculate_sma(prices, 20)
    assert bundle["sma_50"] == IndicatorCalculator.calculate_sma(prices, 50)


def test_sma_bundle_handles_short_series_with_nan() -> None:
    prices = [1.0, 2.0, 3.0]
    bundle = IndicatorCalculator.calculate_sma_bundle(prices)
    assert math.isnan(bundle["sma_10"])
    assert math.isnan(bundle["sma_20"])
    assert math.isnan(bundle["sma_50"])


def test_invalid_period_raises_value_error_for_non_sma_functions() -> None:
    prices = [1.0, 2.0, 3.0]
    try:
        IndicatorCalculator.calculate_ema(prices, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for period=-1 in EMA")

    try:
        IndicatorCalculator.calculate_rsi(prices, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for period=0 in RSI")

    try:
        IndicatorCalculator.calculate_volatility(prices, -2)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for period=-2 in volatility")
