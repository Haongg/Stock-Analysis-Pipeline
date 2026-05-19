from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.flink.es_sink import (
    DualIndexElasticsearchSink,
    ElasticsearchSinkConfig,
    build_daily_index_name,
    check_cluster_health,
    create_elasticsearch_client,
    ensure_elasticsearch_indices,
    setup_elasticsearch_lifecycle,
)


def feature_record() -> dict:
    return {
        "ticker": "ESVALID",
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


def prediction_record() -> dict:
    return {
        "ticker": "ESVALID",
        "prediction_date": "2025-01-16",
        "@timestamp": "2025-01-15T10:30:00Z",
        "predicted_close": 102.25,
        "actual_close": 101.0,
        "model_version": "validation-smoke",
        "confidence": 0.9,
        "prediction_error": None,
        "error_type": None,
    }


def hit_count(response: dict) -> int:
    total = response["hits"]["total"]
    if isinstance(total, dict):
        return int(total["value"])
    return int(total)


def main() -> int:
    config = ElasticsearchSinkConfig(batch_size=2)
    client = create_elasticsearch_client(config)

    health = check_cluster_health(client, config=config)
    print(f"cluster_health={health.get('status')}")

    lifecycle = setup_elasticsearch_lifecycle(client, config=config)
    print(f"lifecycle={lifecycle}")

    index_results = ensure_elasticsearch_indices(client, config=config)
    print(f"indices={index_results}")

    feature = feature_record()
    prediction = prediction_record()
    sink = DualIndexElasticsearchSink(config=config, client=client)
    sink.invoke(feature)
    sink.invoke(prediction)
    result = sink.flush()
    print(f"flush={result}")

    feature_index = build_daily_index_name(config.feature_index, feature)
    prediction_index = build_daily_index_name(config.prediction_index, prediction)
    client.indices.refresh(index=[feature_index, prediction_index])

    feature_hits = hit_count(client.search(index=feature_index, query={"term": {"ticker": "ESVALID"}}))
    prediction_hits = hit_count(client.search(index=prediction_index, query={"term": {"ticker": "ESVALID"}}))
    print(f"feature_index={feature_index} hits={feature_hits}")
    print(f"prediction_index={prediction_index} hits={prediction_hits}")

    if feature_hits < 1 or prediction_hits < 1:
        print("Elasticsearch sink validation failed: expected records were not queryable.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
