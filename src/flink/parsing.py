import json
from datetime import datetime, timezone
from typing import Any


REQUIRED_FIELDS = ("ticker", "date", "open", "high", "low", "close", "volume")


def _parse_iso_to_epoch_ms(value: str) -> int:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_ohlcv_event(raw: str) -> dict[str, Any] | None:
    """
    Parse Kafka raw JSON string into normalized OHLCV event dict.
    Returns None for malformed/invalid records.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, dict):
        return None

    for field in REQUIRED_FIELDS:
        if field not in payload:
            return None

    ticker = str(payload["ticker"]).strip().upper()
    if not ticker:
        return None

    date_str = str(payload["date"]).strip()
    if not date_str:
        return None

    try:
        event_ts_ms = _parse_iso_to_epoch_ms(date_str)
    except ValueError:
        return None

    try:
        parsed = {
            "ticker": ticker,
            "date": date_str,
            "event_ts_ms": event_ts_ms,
            "open": float(payload["open"]),
            "high": float(payload["high"]),
            "low": float(payload["low"]),
            "close": float(payload["close"]),
            "volume": float(payload["volume"]),
        }
    except (TypeError, ValueError):
        return None

    ingested_at = payload.get("ingested_at")
    if ingested_at is not None:
        parsed["ingested_at"] = str(ingested_at)

    return parsed
