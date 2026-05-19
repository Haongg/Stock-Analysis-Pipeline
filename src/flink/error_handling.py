from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def _record_timestamp(record: dict) -> Any:
    return record.get("@timestamp") or record.get("date") or record.get("window_end_ms") or record.get("event_ts_ms")


def _base_prediction_record(record: dict, model_version: str | None = None) -> dict:
    return {
        "ticker": record.get("ticker"),
        "date": record.get("date"),
        "@timestamp": _record_timestamp(record),
        "actual_close": record.get("actual_close", record.get("close")),
        "predicted_close": None,
        "confidence": None,
        "model_version": model_version or record.get("model_version"),
        "prediction_error": None,
        "error_type": None,
        "sequence_ready": bool(record.get("sequence_ready")),
    }


def build_null_prediction(record: dict, reason: str, model_version: str | None = None, error_type: str = "inference_error") -> dict:
    output = _base_prediction_record(record, model_version=model_version)
    output["prediction_error"] = reason
    output["error_type"] = error_type
    return output


def mark_low_confidence(record: dict, reason: str = "sequence_incomplete", model_version: str | None = None) -> dict:
    output = build_null_prediction(record, reason=reason, model_version=model_version, error_type="low_confidence")
    output["confidence"] = 0.0
    return output


def should_skip_inference(record: dict) -> bool:
    return not bool(record.get("sequence_ready"))


def retry_with_backoff(
    fn: Callable[[], Any],
    retries: int = 3,
    base_delay_seconds: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Any:
    if retries <= 0:
        raise ValueError("retries must be > 0")

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            sleep_fn(base_delay_seconds * (2 ** attempt))

    raise last_error  # type: ignore[misc]


def keep_previous_model(current_model: Any, candidate_model: Any) -> Any:
    return candidate_model if candidate_model is not None else current_model


def _normalize_prediction_result(result: Any) -> tuple[Any, float | None]:
    if isinstance(result, dict):
        return result.get("predicted_close", result.get("prediction")), result.get("confidence")
    if isinstance(result, (tuple, list)):
        if len(result) >= 2:
            return result[0], result[1]
        if len(result) == 1:
            return result[0], None
    return result, None


def safe_inference_record(
    record: dict,
    predict_fn: Callable[[dict], Any],
    model_version: str | None = None,
    retries: int = 3,
    base_delay_seconds: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    if should_skip_inference(record):
        return mark_low_confidence(record, model_version=model_version)

    try:
        result = retry_with_backoff(
            lambda: predict_fn(record),
            retries=retries,
            base_delay_seconds=base_delay_seconds,
            sleep_fn=sleep_fn,
        )
    except Exception as exc:
        return build_null_prediction(record, reason=str(exc), model_version=model_version)

    predicted_close, confidence = _normalize_prediction_result(result)
    output = _base_prediction_record(record, model_version=model_version)
    output["predicted_close"] = predicted_close
    output["confidence"] = confidence
    output["prediction_error"] = None
    output["error_type"] = None
    return output
