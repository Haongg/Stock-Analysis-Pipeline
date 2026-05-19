from src.flink.error_handling import (
    build_null_prediction,
    keep_previous_model,
    mark_low_confidence,
    retry_with_backoff,
    safe_inference_record,
    should_skip_inference,
)
from src.flink.monitoring import prediction_has_error


def _record(sequence_ready: bool = True) -> dict:
    return {
        "ticker": "AAPL",
        "date": "2025-01-15",
        "@timestamp": "2025-01-15T16:00:00Z",
        "close": 154.0,
        "sequence_ready": sequence_ready,
    }


def test_incomplete_sequence_returns_low_confidence_null_prediction() -> None:
    output = safe_inference_record(_record(sequence_ready=False), lambda record: 150.0, model_version="v1")

    assert should_skip_inference(_record(sequence_ready=False)) is True
    assert output["predicted_close"] is None
    assert output["confidence"] == 0.0
    assert output["model_version"] == "v1"
    assert output["prediction_error"] == "sequence_incomplete"
    assert output["error_type"] == "low_confidence"
    assert prediction_has_error(output) is True


def test_build_null_prediction_contains_structured_error_fields() -> None:
    output = build_null_prediction(_record(), reason="model timeout", model_version="v2")

    assert output["ticker"] == "AAPL"
    assert output["actual_close"] == 154.0
    assert output["predicted_close"] is None
    assert output["confidence"] is None
    assert output["model_version"] == "v2"
    assert output["prediction_error"] == "model timeout"
    assert output["error_type"] == "inference_error"


def test_mark_low_confidence_uses_zero_confidence() -> None:
    output = mark_low_confidence(_record(sequence_ready=False))

    assert output["predicted_close"] is None
    assert output["confidence"] == 0.0
    assert output["prediction_error"] == "sequence_incomplete"


def test_successful_prediction_returns_prediction_and_clears_error() -> None:
    output = safe_inference_record(
        _record(),
        lambda record: {"predicted_close": 150.75, "confidence": 0.92},
        model_version="v3",
    )

    assert output["predicted_close"] == 150.75
    assert output["confidence"] == 0.92
    assert output["model_version"] == "v3"
    assert output["prediction_error"] is None
    assert output["error_type"] is None
    assert prediction_has_error(output) is False


def test_failed_prediction_returns_null_prediction_with_error_reason() -> None:
    def fail(record: dict) -> float:
        raise RuntimeError("onnx runtime failed")

    output = safe_inference_record(_record(), fail, model_version="v4", sleep_fn=lambda delay: None)

    assert output["predicted_close"] is None
    assert output["confidence"] is None
    assert output["prediction_error"] == "onnx runtime failed"
    assert output["error_type"] == "inference_error"


def test_retry_with_backoff_succeeds_after_transient_failures() -> None:
    calls = {"count": 0}
    delays: list[float] = []

    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("temporary")
        return "ok"

    result = retry_with_backoff(flaky, retries=3, base_delay_seconds=1.0, sleep_fn=delays.append)

    assert result == "ok"
    assert calls["count"] == 3
    assert delays == [1.0, 2.0]


def test_retry_with_backoff_raises_after_failed_attempts() -> None:
    calls = {"count": 0}
    delays: list[float] = []

    def always_fail() -> None:
        calls["count"] += 1
        raise RuntimeError("still failing")

    try:
        retry_with_backoff(always_fail, retries=3, base_delay_seconds=0.5, sleep_fn=delays.append)
    except RuntimeError as exc:
        assert str(exc) == "still failing"
    else:
        raise AssertionError("Expected RuntimeError after retries are exhausted")

    assert calls["count"] == 3
    assert delays == [0.5, 1.0]


def test_keep_previous_model_preserves_current_when_candidate_missing() -> None:
    current = object()
    candidate = object()

    assert keep_previous_model(current, None) is current
    assert keep_previous_model(current, candidate) is candidate
