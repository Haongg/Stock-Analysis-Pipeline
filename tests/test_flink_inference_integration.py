from datetime import datetime, timezone

from src.flink.es_sink import route_document_type
from src.flink.inference_integration import (
    LSTMInferenceMapFunction,
    build_model_not_loaded_record,
    format_prediction_output,
    run_lstm_inference,
)
from src.flink.model_loader import LoadedModel


class _FakePredictor:
    def __init__(self, result=150.75, fail: bool = False):
        self.result = result
        self.fail = fail
        self.calls = []

    def predict_record(self, record: dict):
        self.calls.append(record)
        if self.fail:
            raise RuntimeError("onnx runtime failed")
        return self.result


class _FakeModelLoader:
    def __init__(self, loaded_model=None, fail: bool = False):
        self.loaded_model = loaded_model
        self.fail = fail
        self.open_called = False

    def open(self, runtime_context=None):
        self.open_called = True
        if self.fail:
            raise RuntimeError("model load failed")

    def get_loaded_model(self):
        return self.loaded_model


class _FakeReloadManager:
    def __init__(self, initial_model=None, reload_model=None, fail_reload: bool = False):
        self.initial_model = initial_model
        self.reload_model = reload_model
        self.fail_reload = fail_reload
        self.current_model = initial_model
        self.last_error = None
        self.open_called = False
        self.reload_calls = 0

    def load_initial(self):
        self.open_called = True
        self.current_model = self.initial_model
        return self.current_model

    def maybe_reload(self):
        self.reload_calls += 1
        if self.fail_reload:
            self.last_error = "reload failed"
            return self.current_model
        if self.reload_model is not None:
            self.current_model = self.reload_model
        self.last_error = None
        return self.current_model


class _Counter:
    def __init__(self):
        self.value = 0

    def inc(self, amount: int = 1):
        self.value += amount


class _MetricGroup:
    def __init__(self):
        self.counters = {}

    def counter(self, name: str):
        self.counters[name] = _Counter()
        return self.counters[name]

    def gauge(self, name: str, fn):
        return None


class _RuntimeContext:
    def __init__(self):
        self.metric_group = _MetricGroup()

    def get_metric_group(self):
        return self.metric_group


def _record(sequence_ready: bool = True) -> dict:
    return {
        "ticker": "AAPL",
        "date": "2025-01-15",
        "@timestamp": "2025-01-15T16:00:00Z",
        "close": 154.0,
        "sequence_ready": sequence_ready,
        "feature_sequence": [{"close": 154.0}],
        "feature_columns": ["close"],
    }


def _canonical_prediction_keys() -> set[str]:
    return {
        "type",
        "ticker",
        "date",
        "prediction_date",
        "@timestamp",
        "actual_close",
        "predicted_close",
        "model_version",
        "confidence",
        "inference_time_ms",
        "prediction_error",
        "error_type",
        "sequence_ready",
    }


def _loaded_model(predictor=None, version: str = "v1") -> LoadedModel:
    return LoadedModel(
        predictor=predictor or _FakePredictor(),
        version=version,
        model_path="model.onnx",
        scaler_path="scaler.pkl",
        loaded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_ready_sequence_calls_predictor_and_returns_prediction() -> None:
    predictor = _FakePredictor(result=151.25)
    ticks = iter([10.0, 10.045])

    output = run_lstm_inference(_record(sequence_ready=True), predictor, model_version="v9", now_fn=lambda: next(ticks))

    assert output["type"] == "prediction"
    assert output["predicted_close"] == 151.25
    assert set(output).issuperset(_canonical_prediction_keys())
    assert output["actual_close"] == 154.0
    assert output["prediction_date"] == "2025-01-16T00:00:00Z"
    assert output["model_version"] == "v9"
    assert output["prediction_error"] is None
    assert abs(output["inference_time_ms"] - 45.0) < 1e-9
    assert predictor.calls
    assert route_document_type(output) == "prediction"


def test_incomplete_sequence_returns_low_confidence_prediction_record() -> None:
    predictor = _FakePredictor(result=151.25)
    output = run_lstm_inference(_record(sequence_ready=False), predictor, model_version="v2", now_fn=lambda: 1.0)

    assert output["type"] == "prediction"
    assert set(output).issuperset(_canonical_prediction_keys())
    assert output["predicted_close"] is None
    assert output["confidence"] == 0.0
    assert output["prediction_error"] == "sequence_incomplete"
    assert output["error_type"] == "low_confidence"
    assert predictor.calls == []


def test_predictor_exception_returns_structured_error_record() -> None:
    output = run_lstm_inference(
        _record(sequence_ready=True),
        _FakePredictor(fail=True),
        model_version="v3",
        now_fn=lambda: 2.0,
    )

    assert output["type"] == "prediction"
    assert set(output).issuperset(_canonical_prediction_keys())
    assert output["predicted_close"] is None
    assert output["prediction_error"] == "onnx runtime failed"
    assert output["error_type"] == "inference_error"


def test_map_function_with_loaded_model_attaches_version_and_records_metrics() -> None:
    runtime_context = _RuntimeContext()
    map_fn = LSTMInferenceMapFunction(
        model_loader=_FakeModelLoader(_loaded_model(_FakePredictor(result=200.0), version="v_loaded")),
        now_fn=lambda: 3.0,
    )

    map_fn.open(runtime_context)
    output = map_fn.map(_record(sequence_ready=True))

    assert output["predicted_close"] == 200.0
    assert output["model_version"] == "v_loaded"
    assert runtime_context.metric_group.counters["predictions_total"].value == 1


def test_map_function_uses_reloaded_model_version() -> None:
    initial = _loaded_model(_FakePredictor(result=100.0), version="v1")
    reloaded = _loaded_model(_FakePredictor(result=250.0), version="v2")
    reload_manager = _FakeReloadManager(initial_model=initial, reload_model=reloaded)
    map_fn = LSTMInferenceMapFunction(reload_manager=reload_manager, now_fn=lambda: 4.0)

    map_fn.open()
    output = map_fn.map(_record(sequence_ready=True))

    assert reload_manager.open_called is True
    assert reload_manager.reload_calls == 1
    assert output["predicted_close"] == 250.0
    assert output["model_version"] == "v2"


def test_failed_reload_keeps_previous_model_and_still_predicts() -> None:
    initial = _loaded_model(_FakePredictor(result=175.0), version="v1")
    reload_manager = _FakeReloadManager(initial_model=initial, reload_model=None, fail_reload=True)
    map_fn = LSTMInferenceMapFunction(reload_manager=reload_manager, now_fn=lambda: 5.0)

    map_fn.open()
    output = map_fn.map(_record(sequence_ready=True))

    assert output["predicted_close"] == 175.0
    assert output["model_version"] == "v1"
    assert map_fn.model_load_error == "reload failed"


def test_model_load_failure_emits_model_not_loaded_without_raising() -> None:
    map_fn = LSTMInferenceMapFunction(model_loader=_FakeModelLoader(fail=True))

    map_fn.open()
    output = map_fn.map(_record(sequence_ready=True))

    assert output == build_model_not_loaded_record(_record(sequence_ready=True), reason="model_not_loaded")
    assert output["type"] == "prediction"
    assert set(output).issuperset(_canonical_prediction_keys())
    assert output["prediction_error"] == "model_not_loaded"
    assert output["error_type"] == "model_load_error"


def test_format_prediction_output_preserves_existing_prediction_date() -> None:
    output = format_prediction_output(
        {
            **_record(),
            "prediction_date": "2025-02-01T00:00:00Z",
            "predicted_close": 155.0,
            "inference_time_ms": 12,
        }
    )

    assert output["prediction_date"] == "2025-02-01T00:00:00Z"
    assert output["predicted_close"] == 155.0


def test_format_prediction_output_falls_back_for_unparseable_date() -> None:
    output = format_prediction_output({"ticker": "AAPL", "date": "not-a-date"})

    assert output["prediction_date"] == "not-a-date"
    assert output["@timestamp"] == "not-a-date"
