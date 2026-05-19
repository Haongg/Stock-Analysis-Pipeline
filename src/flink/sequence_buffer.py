from __future__ import annotations

from datetime import date, datetime, timedelta, timezone


FEATURE_SEQUENCE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "volatility",
    "daily_return",
    "close_lag_1",
    "close_lag_5",
    "is_imputed",
]


def parse_record_date(record: dict) -> date:
    raw_value = (
        record.get("@timestamp")
        or record.get("date")
        or record.get("window_end")
        or record.get("window_end_ms")
        or record.get("event_ts_ms")
        or record.get("last_event_ts_ms")
    )
    if raw_value is None:
        raise ValueError("Feature record must include date, @timestamp, window_end_ms, or event_ts_ms.")

    if isinstance(raw_value, datetime):
        return raw_value.astimezone(timezone.utc).date() if raw_value.tzinfo else raw_value.date()
    if isinstance(raw_value, date):
        return raw_value
    if isinstance(raw_value, (int, float)):
        timestamp = float(raw_value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000.0
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).date()

    text = str(raw_value).strip()
    if text.isdigit():
        timestamp = float(text)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000.0
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).date()

    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError as exc:
        raise ValueError(f"Unable to parse feature record date: {raw_value}") from exc


def feature_vector_from_record(record: dict, record_date: date | None = None, is_imputed: int | None = None) -> dict:
    vector_date = record_date or parse_record_date(record)
    vector = {
        "ticker": record.get("ticker"),
        "date": vector_date.isoformat(),
        "event_ts_ms": int(record.get("event_ts_ms") or record.get("last_event_ts_ms") or record.get("window_end_ms") or 0),
    }
    for column in FEATURE_SEQUENCE_COLUMNS:
        if column == "is_imputed":
            value = record.get(column, 0 if is_imputed is None else is_imputed)
            vector[column] = int(value)
        else:
            vector[column] = float(record[column]) if column in record and record[column] is not None else float("nan")
    return vector


def forward_fill_vector(previous_vector: dict, fill_date: date) -> dict:
    filled = dict(previous_vector)
    filled["date"] = fill_date.isoformat()
    filled["is_imputed"] = 1
    return filled


def feature_vector_sort_key(vector: dict) -> tuple[str, int]:
    return (str(vector["date"]), int(vector.get("event_ts_ms", 0)))


def append_feature_vector_to_buffer(buffer: list[dict], record: dict, max_length: int) -> list[dict]:
    record_date = parse_record_date(record)
    new_vector = feature_vector_from_record(record, record_date=record_date, is_imputed=record.get("is_imputed", 0))
    normalized_buffer = sorted(buffer, key=feature_vector_sort_key)

    if normalized_buffer:
        latest_date = parse_record_date(normalized_buffer[-1])
        if record_date > latest_date:
            fill_date = latest_date + timedelta(days=1)
            while fill_date < record_date:
                normalized_buffer.append(forward_fill_vector(normalized_buffer[-1], fill_date))
                fill_date += timedelta(days=1)

    by_date = {item["date"]: item for item in normalized_buffer}
    existing = by_date.get(new_vector["date"])
    if existing is None or int(new_vector.get("event_ts_ms", 0)) >= int(existing.get("event_ts_ms", 0)):
        by_date[new_vector["date"]] = new_vector

    deduped = sorted(by_date.values(), key=feature_vector_sort_key)
    return deduped[-max_length:]


def with_sequence_metadata(record: dict, buffer: list[dict], sequence_length: int) -> dict:
    ready = len(buffer) >= sequence_length
    sequence = buffer[-sequence_length:] if ready else []
    enriched = dict(record)
    enriched.update(
        {
            "sequence_ready": ready,
            "sequence_length": min(len(buffer), sequence_length),
            "feature_sequence": sequence,
            "feature_columns": list(FEATURE_SEQUENCE_COLUMNS),
        }
    )
    return enriched
