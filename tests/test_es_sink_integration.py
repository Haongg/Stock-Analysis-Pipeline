import os

import pytest

from src.flink.es_sink import (
    DualIndexElasticsearchSink,
    ElasticsearchSinkConfig,
    build_daily_index_name,
    check_cluster_health,
    create_elasticsearch_client,
    ensure_elasticsearch_indices,
    setup_elasticsearch_lifecycle,
)


pytestmark = pytest.mark.skipif(
    os.getenv("ES_INTEGRATION_TEST") != "1",
    reason="Set ES_INTEGRATION_TEST=1 to run real Elasticsearch integration tests.",
)


def _feature_record() -> dict:
    return {
        "ticker": "ITEST",
        "date": "2025-01-15",
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 1000000,
        "sma_10": 100.5,
        "sma_20": 100.1,
        "sma_50": 99.8,
        "ema_20": 100.2,
        "rsi": 55.0,
        "macd": 1.2,
        "macd_signal": 1.0,
        "macd_hist": 0.2,
        "volatility": 0.015,
        "daily_return": 0.01,
        "close_lag_1": 100.0,
        "close_lag_5": 98.0,
        "is_imputed": 0,
    }


def _prediction_record() -> dict:
    return {
        "ticker": "ITEST",
        "prediction_date": "2025-01-16",
        "@timestamp": "2025-01-15T10:30:00Z",
        "predicted_close": 102.25,
        "actual_close": 101.0,
        "model_version": "integration-test",
        "confidence": 0.9,
        "prediction_error": None,
        "error_type": None,
    }


def _hit_count(response: dict) -> int:
    total = response["hits"]["total"]
    if isinstance(total, dict):
        return int(total["value"])
    return int(total)


def test_dual_index_sink_writes_and_queries_real_elasticsearch() -> None:
    config = ElasticsearchSinkConfig(batch_size=2)
    client = create_elasticsearch_client(config)

    health = check_cluster_health(client, config=config)
    assert health["status"] in {"green", "yellow"}

    setup_elasticsearch_lifecycle(client, config=config)
    ensure_elasticsearch_indices(client, config=config)

    feature = _feature_record()
    prediction = _prediction_record()
    sink = DualIndexElasticsearchSink(config=config, client=client)
    sink.invoke(feature)
    sink.invoke(prediction)
    sink.flush()

    feature_index = build_daily_index_name(config.feature_index, feature)
    prediction_index = build_daily_index_name(config.prediction_index, prediction)
    client.indices.refresh(index=[feature_index, prediction_index])

    feature_hits = client.search(index=feature_index, query={"term": {"ticker": "ITEST"}})
    prediction_hits = client.search(index=prediction_index, query={"term": {"ticker": "ITEST"}})

    assert _hit_count(feature_hits) >= 1
    assert _hit_count(prediction_hits) >= 1
