from src.flink.main import _accumulate_ohlcv_event, _init_ohlcv_accumulator


def _event(ts: int, o: float, h: float, l: float, c: float, v: float) -> dict:
    return {
        "ticker": "AAPL",
        "event_ts_ms": ts,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "date": "2025-01-15T00:00:00Z",
    }


def test_ohlcv_aggregate_multi_event_same_ticker() -> None:
    acc = _init_ohlcv_accumulator()
    acc = _accumulate_ohlcv_event(acc, _event(1000, 10, 15, 9, 14, 100))
    acc = _accumulate_ohlcv_event(acc, _event(2000, 14, 18, 13, 17, 200))
    acc = _accumulate_ohlcv_event(acc, _event(3000, 17, 19, 16, 18, 50))

    assert acc["open"] == 10.0
    assert acc["high"] == 19.0
    assert acc["low"] == 9.0
    assert acc["close"] == 18.0
    assert acc["volume_sum"] == 350.0
    assert acc["event_count"] == 3
    assert acc["min_price"] == 9.0
    assert acc["max_price"] == 19.0
    assert acc["last_event_ts_ms"] == 3000


def test_ohlcv_aggregate_out_of_order_timestamps() -> None:
    acc = _init_ohlcv_accumulator()
    acc = _accumulate_ohlcv_event(acc, _event(3000, 30, 35, 29, 34, 300))
    acc = _accumulate_ohlcv_event(acc, _event(1000, 10, 12, 8, 11, 100))
    acc = _accumulate_ohlcv_event(acc, _event(2000, 20, 22, 19, 21, 200))

    assert acc["open"] == 10.0
    assert acc["close"] == 34.0
    assert acc["high"] == 35.0
    assert acc["low"] == 8.0
    assert acc["volume_sum"] == 600.0


def test_ohlcv_aggregate_same_timestamp_duplicate_uses_last_close() -> None:
    acc = _init_ohlcv_accumulator()
    acc = _accumulate_ohlcv_event(acc, _event(1000, 10, 11, 9, 10.5, 100))
    acc = _accumulate_ohlcv_event(acc, _event(1000, 10.2, 12, 9.5, 11.2, 50))

    assert acc["open"] == 10.0
    assert acc["close"] == 11.2
    assert acc["event_count"] == 2
    assert acc["volume_sum"] == 150.0


def test_ohlcv_aggregate_single_event_window() -> None:
    acc = _init_ohlcv_accumulator()
    acc = _accumulate_ohlcv_event(acc, _event(5000, 100, 105, 99, 103, 999))

    assert acc["open"] == 100.0
    assert acc["high"] == 105.0
    assert acc["low"] == 99.0
    assert acc["close"] == 103.0
    assert acc["volume_sum"] == 999.0
    assert acc["event_count"] == 1
