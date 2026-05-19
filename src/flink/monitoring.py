from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any


LATENCY_UNDER_100MS = "under_100ms"
LATENCY_100_TO_500MS = "100_500ms"
LATENCY_OVER_500MS = "over_500ms"


def classify_latency_bucket(inference_time_ms: float | int | None) -> str | None:
    if inference_time_ms is None:
        return None
    latency = float(inference_time_ms)
    if latency < 100.0:
        return LATENCY_UNDER_100MS
    if latency <= 500.0:
        return LATENCY_100_TO_500MS
    return LATENCY_OVER_500MS


def parse_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def model_age_days(training_timestamp: Any, now: datetime | None = None) -> float | None:
    trained_at = parse_utc_datetime(training_timestamp)
    if trained_at is None:
        return None

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    age_seconds = (current_time - trained_at).total_seconds()
    if age_seconds < 0:
        return 0.0
    return age_seconds / 86400.0


def prediction_has_error(record: dict) -> bool:
    error = record.get("prediction_error")
    if error is None:
        return False
    if isinstance(error, bool):
        return error
    return bool(str(error).strip())


def count_imputed_sequence_rows(record: dict) -> int:
    sequence = record.get("feature_sequence") or []
    return sum(1 for item in sequence if int(item.get("is_imputed", 0)) == 1)


def sequence_metric_classification(record: dict) -> dict[str, int]:
    ready = bool(record.get("sequence_ready"))
    return {
        "records": 1,
        "ready": 1 if ready else 0,
        "not_ready": 0 if ready else 1,
        "imputed_rows": count_imputed_sequence_rows(record),
    }


class InferenceMetricsRecorder:
    """Reusable metric recorder for model inference operators."""

    def __init__(self, metric_group, model_training_timestamp: str | None = None) -> None:
        self.predictions_total = metric_group.counter("predictions_total")
        self.prediction_errors_total = metric_group.counter("prediction_errors_total")
        self.predictions_latency_under_100ms_total = metric_group.counter(
            f"predictions_latency_{LATENCY_UNDER_100MS}_total"
        )
        self.predictions_latency_100_500ms_total = metric_group.counter(
            f"predictions_latency_{LATENCY_100_TO_500MS}_total"
        )
        self.predictions_latency_over_500ms_total = metric_group.counter(
            f"predictions_latency_{LATENCY_OVER_500MS}_total"
        )
        self.model_age_days_value = model_age_days(model_training_timestamp or os.getenv("MODEL_TRAINING_TIMESTAMP"))
        try:
            metric_group.gauge("model_age_days", lambda: self.model_age_days_value or 0.0)
        except AttributeError:
            pass

    def record(self, record: dict) -> None:
        self.predictions_total.inc()
        if prediction_has_error(record):
            self.prediction_errors_total.inc()

        bucket = classify_latency_bucket(record.get("inference_time_ms"))
        if bucket == LATENCY_UNDER_100MS:
            self.predictions_latency_under_100ms_total.inc()
        elif bucket == LATENCY_100_TO_500MS:
            self.predictions_latency_100_500ms_total.inc()
        elif bucket == LATENCY_OVER_500MS:
            self.predictions_latency_over_500ms_total.inc()

        age = model_age_days(record.get("model_training_timestamp") or os.getenv("MODEL_TRAINING_TIMESTAMP"))
        if age is not None:
            self.model_age_days_value = age
