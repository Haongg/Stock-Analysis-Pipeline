from datetime import datetime, timezone

from src.flink.monitoring import (
    LATENCY_100_TO_500MS,
    LATENCY_OVER_500MS,
    LATENCY_UNDER_100MS,
    classify_latency_bucket,
    count_imputed_sequence_rows,
    model_age_days,
    prediction_has_error,
    sequence_metric_classification,
)


def test_classify_latency_bucket() -> None:
    assert classify_latency_bucket(99.9) == LATENCY_UNDER_100MS
    assert classify_latency_bucket(100) == LATENCY_100_TO_500MS
    assert classify_latency_bucket(500) == LATENCY_100_TO_500MS
    assert classify_latency_bucket(501) == LATENCY_OVER_500MS
    assert classify_latency_bucket(None) is None


def test_model_age_days_from_iso_timestamp() -> None:
    now = datetime(2025, 1, 3, 12, 0, tzinfo=timezone.utc)
    assert model_age_days("2025-01-01T12:00:00Z", now=now) == 2.0


def test_model_age_days_invalid_or_missing_timestamp() -> None:
    assert model_age_days(None) is None
    assert model_age_days("") is None
    assert model_age_days("not-a-date") is None


def test_prediction_has_error() -> None:
    assert prediction_has_error({}) is False
    assert prediction_has_error({"prediction_error": None}) is False
    assert prediction_has_error({"prediction_error": False}) is False
    assert prediction_has_error({"prediction_error": "model timeout"}) is True
    assert prediction_has_error({"prediction_error": True}) is True


def test_sequence_metric_classification_ready_and_imputed_rows() -> None:
    record = {
        "sequence_ready": True,
        "feature_sequence": [
            {"is_imputed": 0},
            {"is_imputed": 1},
            {"is_imputed": 1},
        ],
    }

    assert count_imputed_sequence_rows(record) == 2
    assert sequence_metric_classification(record) == {
        "records": 1,
        "ready": 1,
        "not_ready": 0,
        "imputed_rows": 2,
    }


def test_sequence_metric_classification_not_ready() -> None:
    assert sequence_metric_classification({"sequence_ready": False}) == {
        "records": 1,
        "ready": 0,
        "not_ready": 1,
        "imputed_rows": 0,
    }
