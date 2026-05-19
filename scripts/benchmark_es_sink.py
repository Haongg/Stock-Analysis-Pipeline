from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.flink.es_sink import (
    ElasticsearchSinkConfig,
    ElasticsearchSinkFunction,
    create_elasticsearch_client,
    ensure_elasticsearch_indices,
    setup_elasticsearch_lifecycle,
)


class Counter:
    def __init__(self) -> None:
        self.value = 0

    def inc(self, amount=1) -> None:
        self.value += amount


class MetricGroup:
    def __init__(self) -> None:
        self.counters: dict[str, Counter] = {}

    def counter(self, name: str) -> Counter:
        self.counters.setdefault(name, Counter())
        return self.counters[name]


def synthetic_feature_record(position: int) -> dict:
    close = 100.0 + (position % 50) * 0.1
    return {
        "ticker": f"BENCH{position % 10}",
        "date": "2025-01-15",
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000000 + position,
        "sma_10": close - 0.1,
        "sma_20": close - 0.2,
        "sma_50": close - 0.5,
        "ema_20": close - 0.15,
        "rsi": 55.0,
        "macd": 1.2,
        "macd_signal": 1.0,
        "macd_hist": 0.2,
        "volatility": 0.015,
        "daily_return": 0.001,
        "close_lag_1": close - 0.2,
        "close_lag_5": close - 0.6,
        "is_imputed": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Elasticsearch sink throughput with synthetic feature docs.")
    parser.add_argument("--events", type=int, default=1000, help="Number of synthetic events to index.")
    parser.add_argument("--batch-size", type=int, default=100, help="Bulk batch size.")
    parser.add_argument("--target-events-per-sec", type=float, default=1000.0, help="Throughput target.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ElasticsearchSinkConfig(batch_size=args.batch_size)
    client = create_elasticsearch_client(config)

    setup_elasticsearch_lifecycle(client, config=config)
    ensure_elasticsearch_indices(client, config=config)

    metric_group = MetricGroup()
    sink = ElasticsearchSinkFunction(
        index_name=config.feature_index,
        config=config,
        client=client,
        metric_group=metric_group,
    )

    started = time.perf_counter()
    for position in range(args.events):
        sink.invoke(synthetic_feature_record(position))
    sink.close()
    elapsed = max(time.perf_counter() - started, 1e-9)
    throughput = args.events / elapsed

    inserted = metric_group.counters.get("documents_inserted", Counter()).value
    errors = metric_group.counters.get("insert_errors", Counter()).value
    print(f"events={args.events}")
    print(f"elapsed_seconds={elapsed:.4f}")
    print(f"events_per_sec={throughput:.2f}")
    print(f"documents_inserted={inserted}")
    print(f"insert_errors={errors}")

    if inserted < args.events or errors:
        print("Benchmark failed: not all documents were inserted successfully.")
        return 1
    if throughput < args.target_events_per_sec:
        print(f"Benchmark below target: {throughput:.2f} < {args.target_events_per_sec:.2f} events/sec")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
