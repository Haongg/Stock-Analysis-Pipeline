from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from src.flink.error_handling import build_null_prediction, safe_inference_record
from src.flink.model_loader import ModelLoaderRichMapFunction, ModelReloadManager
from src.flink.monitoring import InferenceMetricsRecorder


try:
    from pyflink.datastream.functions import RichMapFunction
except Exception:
    class RichMapFunction:  # type: ignore[no-redef]
        """Fallback base class so unit tests do not require PyFlink."""

        pass


def _add_prediction_metadata(record: dict, inference_time_ms: float) -> dict:
    output = dict(record)
    output["type"] = "prediction"
    output["inference_time_ms"] = float(inference_time_ms)
    return output


def _parse_prediction_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000.0
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.isdigit():
            timestamp = float(text)
            if timestamp > 10_000_000_000:
                timestamp = timestamp / 1000.0
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _derive_prediction_date(record: dict) -> Any:
    if record.get("prediction_date") is not None:
        return record.get("prediction_date")

    source_value = record.get("date") or record.get("@timestamp")
    parsed = _parse_prediction_datetime(source_value)
    if parsed is None:
        return source_value
    return _iso_utc(parsed + timedelta(days=1))


def format_prediction_output(record: dict, prediction_date_fn: Callable[[dict], Any] | None = None) -> dict:
    prediction_date = prediction_date_fn(record) if prediction_date_fn else _derive_prediction_date(record)
    output = {
        "type": "prediction",
        "ticker": record.get("ticker"),
        "date": record.get("date"),
        "prediction_date": prediction_date,
        "@timestamp": record.get("@timestamp") or record.get("date") or prediction_date,
        "actual_close": record.get("actual_close", record.get("close")),
        "predicted_close": record.get("predicted_close"),
        "model_version": record.get("model_version"),
        "confidence": record.get("confidence"),
        "inference_time_ms": float(record.get("inference_time_ms") or 0.0),
        "prediction_error": record.get("prediction_error"),
        "error_type": record.get("error_type"),
        "sequence_ready": bool(record.get("sequence_ready")),
    }
    for optional_key in ("model_training_timestamp", "feature_columns", "sequence_length"):
        if optional_key in record:
            output[optional_key] = record[optional_key]
    return output


def run_lstm_inference(
    record: dict,
    predictor: Any,
    model_version: str | None = None,
    now_fn: Callable[[], float] | None = None,
) -> dict:
    """Run safe LSTM inference and return a prediction-style record."""
    clock = now_fn or time.perf_counter
    start = clock()
    output = safe_inference_record(
        record,
        lambda source_record: predictor.predict_record(source_record),
        model_version=model_version,
    )
    elapsed_ms = max((clock() - start) * 1000.0, 0.0)
    return format_prediction_output(_add_prediction_metadata(output, elapsed_ms))


def build_model_not_loaded_record(
    record: dict,
    reason: str = "model_not_loaded",
    model_version: str | None = None,
) -> dict:
    output = build_null_prediction(
        record,
        reason=reason,
        model_version=model_version,
        error_type="model_load_error",
    )
    return format_prediction_output(_add_prediction_metadata(output, 0.0))


class LSTMInferenceMapFunction(RichMapFunction):
    """Map sequenced feature records into structured prediction records."""

    def __init__(
        self,
        model_loader: ModelLoaderRichMapFunction | None = None,
        reload_manager: ModelReloadManager | None = None,
        metrics_recorder_factory: Callable[..., InferenceMetricsRecorder] = InferenceMetricsRecorder,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self.reload_manager = (
            reload_manager
            if reload_manager is not None
            else (None if model_loader is not None else ModelReloadManager())
        )
        self.model_loader = model_loader
        self.metrics_recorder_factory = metrics_recorder_factory
        self.now_fn = now_fn
        self.loaded_model = None
        self.metrics_recorder = None
        self.model_load_error: str | None = None

    def open(self, runtime_context: Any = None) -> None:
        try:
            if self.reload_manager is not None:
                self.loaded_model = self.reload_manager.load_initial()
            else:
                loader = self.model_loader or ModelLoaderRichMapFunction()
                self.model_loader = loader
                loader.open(runtime_context)
                self.loaded_model = loader.get_loaded_model()
            self.model_load_error = None
        except Exception as exc:
            self.loaded_model = None
            self.model_load_error = str(exc) or "model_not_loaded"

        if runtime_context is not None:
            try:
                self.metrics_recorder = self.metrics_recorder_factory(runtime_context.get_metric_group())
            except Exception:
                self.metrics_recorder = None

    def map(self, record: dict) -> dict:
        if self.reload_manager is not None:
            try:
                self.loaded_model = self.reload_manager.maybe_reload()
                self.model_load_error = self.reload_manager.last_error
            except Exception as exc:
                self.model_load_error = str(exc) or "model_reload_failed"

        if self.loaded_model is None:
            output = build_model_not_loaded_record(record, reason="model_not_loaded")
        else:
            output = run_lstm_inference(
                record,
                self.loaded_model.predictor,
                model_version=self.loaded_model.version,
                now_fn=self.now_fn,
            )

        if self.metrics_recorder is not None:
            self.metrics_recorder.record(output)
        return output
