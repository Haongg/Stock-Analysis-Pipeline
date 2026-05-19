import ast
from datetime import datetime, timezone
from pathlib import Path

from src.flink.sequence_buffer import (
    FEATURE_SEQUENCE_COLUMNS,
    append_feature_vector_to_buffer,
    with_sequence_metadata,
)


def _record(day: int, close: float = 100.0, event_ts_ms: int | None = None) -> dict:
    timestamp = datetime(2025, 1, day, tzinfo=timezone.utc)
    value = close + day
    return {
        "ticker": "AAPL",
        "date": timestamp.isoformat(),
        "event_ts_ms": event_ts_ms if event_ts_ms is not None else int(timestamp.timestamp() * 1000),
        "open": value - 1.0,
        "high": value + 2.0,
        "low": value - 2.0,
        "close": value,
        "volume": 1000.0 + day,
        "sma_10": value,
        "sma_20": value,
        "sma_50": value,
        "ema_20": value,
        "rsi": 50.0,
        "macd": 1.0,
        "macd_signal": 0.8,
        "macd_hist": 0.2,
        "volatility": 0.01,
        "daily_return": 0.001,
        "close_lag_1": value - 0.5,
        "close_lag_5": value - 2.5,
    }


def _dataset_base_feature_columns() -> list[str]:
    source = Path("dags/tasks/dataset_creator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "BASE_FEATURE_COLUMNS":
                    return ast.literal_eval(node.value)
    raise AssertionError("BASE_FEATURE_COLUMNS not found in dataset_creator.py")


def test_sequence_buffer_first_append_not_ready() -> None:
    buffer = append_feature_vector_to_buffer([], _record(1), max_length=30)
    enriched = with_sequence_metadata(_record(1), buffer, sequence_length=30)

    assert len(buffer) == 1
    assert enriched["sequence_ready"] is False
    assert enriched["sequence_length"] == 1
    assert enriched["feature_sequence"] == []


def test_sequence_buffer_trims_to_latest_sequence_length() -> None:
    buffer = []
    for day in range(1, 32):
        buffer = append_feature_vector_to_buffer(buffer, _record(day), max_length=30)

    enriched = with_sequence_metadata(_record(31), buffer, sequence_length=30)

    assert len(buffer) == 30
    assert buffer[0]["date"] == "2025-01-02"
    assert buffer[-1]["date"] == "2025-01-31"
    assert enriched["sequence_ready"] is True
    assert len(enriched["feature_sequence"]) == 30


def test_sequence_buffer_deduplicates_same_date_with_latest_event() -> None:
    older = _record(1, close=100.0, event_ts_ms=1000)
    newer = _record(1, close=200.0, event_ts_ms=2000)
    stale = _record(1, close=300.0, event_ts_ms=1500)

    buffer = append_feature_vector_to_buffer([], older, max_length=30)
    buffer = append_feature_vector_to_buffer(buffer, newer, max_length=30)
    buffer = append_feature_vector_to_buffer(buffer, stale, max_length=30)

    assert len(buffer) == 1
    assert buffer[0]["event_ts_ms"] == 2000
    assert buffer[0]["close"] == newer["close"]


def test_sequence_buffer_forward_fills_missing_calendar_days() -> None:
    buffer = append_feature_vector_to_buffer([], _record(1, close=100.0), max_length=30)
    buffer = append_feature_vector_to_buffer(buffer, _record(4, close=200.0), max_length=30)

    assert [item["date"] for item in buffer] == [
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
        "2025-01-04",
    ]
    assert buffer[1]["is_imputed"] == 1
    assert buffer[2]["is_imputed"] == 1
    assert buffer[3]["is_imputed"] == 0
    assert buffer[1]["close"] == buffer[0]["close"]


def test_sequence_feature_columns_match_dataset_creator_order() -> None:
    assert FEATURE_SEQUENCE_COLUMNS == _dataset_base_feature_columns() + ["is_imputed"]
