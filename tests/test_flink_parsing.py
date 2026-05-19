from src.flink.parsing import parse_ohlcv_event


def test_parse_ohlcv_event_valid_z_timestamp() -> None:
    raw = (
        '{"ticker":"aapl","date":"2025-01-15T00:00:00Z","open":100.0,'
        '"high":110.0,"low":95.0,"close":108.0,"volume":12345,"ingested_at":"2025-01-15T00:01:00Z"}'
    )
    event = parse_ohlcv_event(raw)

    assert event is not None
    assert event["ticker"] == "AAPL"
    assert event["event_ts_ms"] > 0
    assert event["open"] == 100.0
    assert event["volume"] == 12345.0
    assert event["ingested_at"] == "2025-01-15T00:01:00Z"


def test_parse_ohlcv_event_valid_offset_timestamp() -> None:
    raw = (
        '{"ticker":"MSFT","date":"2025-01-15T07:00:00+07:00","open":200,'
        '"high":210,"low":190,"close":205,"volume":999}'
    )
    event = parse_ohlcv_event(raw)

    assert event is not None
    assert event["event_ts_ms"] > 0
    assert "ingested_at" not in event


def test_parse_ohlcv_event_missing_required_field() -> None:
    raw = '{"ticker":"AAPL","date":"2025-01-15T00:00:00Z","open":1,"high":2,"low":0,"close":1}'
    assert parse_ohlcv_event(raw) is None


def test_parse_ohlcv_event_invalid_numeric_type() -> None:
    raw = (
        '{"ticker":"AAPL","date":"2025-01-15T00:00:00Z","open":"abc",'
        '"high":2,"low":0,"close":1,"volume":100}'
    )
    assert parse_ohlcv_event(raw) is None


def test_parse_ohlcv_event_invalid_json() -> None:
    raw = '{"ticker":"AAPL","date":"2025-01-15T00:00:00Z",'
    assert parse_ohlcv_event(raw) is None
