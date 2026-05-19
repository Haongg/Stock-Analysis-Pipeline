import math
from datetime import datetime, timedelta, timezone

from src.flink.indicator_enrichment import build_indicator_history, enrich_record_with_indicators
from src.flink.sequence_buffer import FEATURE_SEQUENCE_COLUMNS, append_feature_vector_to_buffer


def _aggregated_record(day: int, close: float, window_end_ms: int | None = None) -> dict:
    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1)
    ts_ms = window_end_ms if window_end_ms is not None else int(timestamp.timestamp() * 1000)
    return {
        "ticker": "AAPL",
        "window_start_ms": ts_ms - 86_400_000,
        "window_end_ms": ts_ms,
        "open": close - 1.0,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": 1000.0 + day,
        "min_price": close - 2.0,
        "max_price": close + 2.0,
        "event_count": 1,
        "last_event_ts_ms": ts_ms,
        "is_partial": False,
    }


def _build_history(days: int, start_close: float = 100.0) -> list[dict]:
    history: list[dict] = []
    for day in range(1, days + 1):
        history = build_indicator_history(history, _aggregated_record(day, start_close + day), max_length=120)
    return history


def test_first_enriched_record_has_feature_schema_with_nan_indicators() -> None:
    record = _aggregated_record(1, 101.0)
    history = build_indicator_history([], record, max_length=120)
    enriched = enrich_record_with_indicators(record, history)

    for column in FEATURE_SEQUENCE_COLUMNS:
        assert column in enriched

    assert enriched["date"] == "2025-01-01T00:00:00Z"
    assert enriched["@timestamp"] == "2025-01-01T00:00:00Z"
    assert math.isnan(enriched["sma_10"])
    assert math.isnan(enriched["ema_20"])
    assert math.isnan(enriched["rsi"])
    assert math.isnan(enriched["macd"])
    assert math.isnan(enriched["volatility"])
    assert math.isnan(enriched["daily_return"])


def test_enriched_record_calculates_all_indicators_after_history_is_ready() -> None:
    history = _build_history(60)
    current = _aggregated_record(60, 160.0)
    enriched = enrich_record_with_indicators(current, history)

    assert enriched["sma_10"] == sum(range(151, 161)) / 10.0
    assert enriched["sma_20"] == sum(range(141, 161)) / 20.0
    assert enriched["sma_50"] == sum(range(111, 161)) / 50.0
    assert not math.isnan(enriched["ema_20"])
    assert not math.isnan(enriched["rsi"])
    assert not math.isnan(enriched["macd"])
    assert not math.isnan(enriched["macd_signal"])
    assert not math.isnan(enriched["macd_hist"])
    assert not math.isnan(enriched["volatility"])
    assert enriched["close_lag_1"] == 159.0
    assert enriched["close_lag_5"] == 155.0
    assert enriched["daily_return"] == (160.0 - 159.0) / 159.0


def test_indicator_history_deduplicates_same_window_with_latest_event() -> None:
    older = _aggregated_record(1, 101.0, window_end_ms=1000)
    older["last_event_ts_ms"] = 1000
    newer = _aggregated_record(1, 202.0, window_end_ms=1000)
    newer["last_event_ts_ms"] = 2000
    stale = _aggregated_record(1, 303.0, window_end_ms=1000)
    stale["last_event_ts_ms"] = 1500

    history = build_indicator_history([], older, max_length=120)
    history = build_indicator_history(history, newer, max_length=120)
    history = build_indicator_history(history, stale, max_length=120)

    assert len(history) == 1
    assert history[0]["close"] == 202.0
    assert history[0]["last_event_ts_ms"] == 2000


def test_zero_previous_close_keeps_nan_return_and_volatility_policy() -> None:
    history = []
    for day, close in [(1, 100.0), (2, 0.0), (3, 110.0)]:
        history = build_indicator_history(history, _aggregated_record(day, close), max_length=120)

    enriched = enrich_record_with_indicators(_aggregated_record(3, 110.0), history)

    assert math.isnan(enriched["daily_return"])
    assert math.isnan(enriched["volatility"])


def test_enriched_record_can_feed_sequence_buffer() -> None:
    history = _build_history(30)
    enriched = enrich_record_with_indicators(_aggregated_record(30, 130.0), history)
    buffer = append_feature_vector_to_buffer([], enriched, max_length=30)

    assert len(buffer) == 1
    assert buffer[0]["ticker"] == "AAPL"
    assert buffer[0]["date"] == "2025-01-30"
    for column in FEATURE_SEQUENCE_COLUMNS:
        assert column in buffer[0]
