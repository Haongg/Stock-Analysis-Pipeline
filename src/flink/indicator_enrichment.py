from __future__ import annotations

from datetime import datetime, timezone

from src.flink.indicators import IndicatorCalculator


def _timestamp_ms_to_iso(timestamp_ms: int | float) -> str:
    return datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _record_timestamp_ms(record: dict) -> int:
    raw_value = (
        record.get("window_end_ms")
        or record.get("last_event_ts_ms")
        or record.get("event_ts_ms")
        or record.get("@timestamp")
        or record.get("date")
    )
    if raw_value is None:
        return 0
    if isinstance(raw_value, (int, float)):
        timestamp = float(raw_value)
        return int(timestamp if timestamp > 10_000_000_000 else timestamp * 1000)
    if isinstance(raw_value, datetime):
        value = raw_value if raw_value.tzinfo else raw_value.replace(tzinfo=timezone.utc)
        return int(value.timestamp() * 1000)

    text = str(raw_value).strip()
    if text.isdigit():
        timestamp = float(text)
        return int(timestamp if timestamp > 10_000_000_000 else timestamp * 1000)
    normalized = text.replace("Z", "+00:00")
    try:
        value = datetime.fromisoformat(normalized)
    except ValueError:
        return 0
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


def _history_key(record: dict) -> str:
    timestamp_ms = _record_timestamp_ms(record)
    ticker = record.get("ticker", "")
    return f"{ticker}:{timestamp_ms}"


def _history_sort_key(record: dict) -> tuple[int, int]:
    timestamp_ms = _record_timestamp_ms(record)
    event_ts_ms = int(record.get("last_event_ts_ms") or record.get("event_ts_ms") or timestamp_ms)
    return timestamp_ms, event_ts_ms


def normalize_feature_record_timestamp(record: dict) -> dict:
    """Ensure aggregated records expose date fields expected by downstream feature code."""
    enriched = dict(record)
    timestamp_ms = _record_timestamp_ms(enriched)
    if timestamp_ms:
        timestamp_iso = _timestamp_ms_to_iso(timestamp_ms)
        enriched.setdefault("date", timestamp_iso)
        enriched.setdefault("@timestamp", timestamp_iso)
        enriched.setdefault("event_ts_ms", timestamp_ms)
    enriched.setdefault("is_imputed", 0)
    return enriched


def build_indicator_history(buffer: list[dict], current_record: dict, max_length: int | None = None) -> list[dict]:
    """Merge current record into close history, dedupe by ticker/window timestamp, and sort oldest first."""
    normalized: dict[str, dict] = {}
    for item in [*buffer, normalize_feature_record_timestamp(current_record)]:
        entry = normalize_feature_record_timestamp(item)
        if "close" not in entry or entry["close"] is None:
            continue
        key = _history_key(entry)
        existing = normalized.get(key)
        if existing is None or _history_sort_key(entry) >= _history_sort_key(existing):
            normalized[key] = {
                "ticker": entry.get("ticker"),
                "date": entry.get("date"),
                "@timestamp": entry.get("@timestamp"),
                "event_ts_ms": int(entry.get("event_ts_ms") or _record_timestamp_ms(entry)),
                "window_end_ms": int(entry.get("window_end_ms") or _record_timestamp_ms(entry)),
                "last_event_ts_ms": int(entry.get("last_event_ts_ms") or entry.get("event_ts_ms") or _record_timestamp_ms(entry)),
                "close": float(entry["close"]),
            }

    history = sorted(normalized.values(), key=_history_sort_key)
    if max_length is not None and len(history) > max_length:
        history = history[-max_length:]
    return history


def enrich_record_with_indicators(record: dict, close_history: list[dict]) -> dict:
    """Add the indicator feature schema consumed by the LSTM sequence buffer."""
    enriched = normalize_feature_record_timestamp(record)
    prices = [float(item["close"]) for item in sorted(close_history, key=_history_sort_key)]
    sma_values = IndicatorCalculator.calculate_sma_bundle(prices)
    macd, macd_signal, macd_hist = IndicatorCalculator.calculate_macd(prices)
    lag_values = IndicatorCalculator.calculate_lag_features(prices)

    enriched.update(sma_values)
    enriched.update(
        {
            "ema_20": IndicatorCalculator.calculate_ema(prices, 20),
            "rsi": IndicatorCalculator.calculate_rsi(prices, 14),
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "volatility": IndicatorCalculator.calculate_volatility(prices, 20),
            **lag_values,
            "is_imputed": int(enriched.get("is_imputed", 0)),
        }
    )
    return enriched
