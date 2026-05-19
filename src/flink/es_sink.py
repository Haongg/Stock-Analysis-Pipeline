from __future__ import annotations

import logging
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger(__name__)

FEATURE_DOCUMENT_TYPE = "features"
PREDICTION_DOCUMENT_TYPE = "prediction"

FEATURE_MARKER_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sma_20",
    "ema_20",
    "rsi",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_histogram",
    "volatility",
    "volatility_20",
    "daily_return",
    "close_lag_1",
    "close_lag_5",
}

PREDICTION_MARKER_FIELDS = {
    "prediction_date",
    "predicted_close",
    "confidence",
    "model_version",
    "prediction_error",
    "error_type",
}

ELASTICSEARCH_DIR = Path(__file__).resolve().parents[1] / "elasticsearch"
INDEX_MAPPING_FILES = {
    FEATURE_DOCUMENT_TYPE: ELASTICSEARCH_DIR / "stock_engineered_features_mapping.json",
    PREDICTION_DOCUMENT_TYPE: ELASTICSEARCH_DIR / "stock_predictions_mapping.json",
}

REQUIRED_FEATURE_MAPPING_FIELDS = {
    "@timestamp",
    "type",
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "volatility",
    "daily_return",
    "close_lag_1",
    "close_lag_5",
    "is_imputed",
}

REQUIRED_PREDICTION_MAPPING_FIELDS = {
    "@timestamp",
    "type",
    "ticker",
    "prediction_date",
    "predicted_close",
    "actual_close",
    "model_version",
    "confidence",
    "prediction_error",
    "error_type",
}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ElasticsearchSinkConfig:
    host: str = os.getenv("ES_HOST", "localhost")
    port: int = _env_int("ES_PORT", 9200)
    scheme: str = os.getenv("ES_SCHEME", "http")
    username: str | None = os.getenv("ES_USER")
    password: str | None = os.getenv("ES_PASSWORD")
    request_timeout: int = _env_int("ES_REQUEST_TIMEOUT_SECONDS", 30)
    batch_size: int = _env_int("ES_BULK_BATCH_SIZE", 100)
    flush_interval_seconds: int = _env_int("ES_BULK_FLUSH_INTERVAL_SECONDS", 5)
    core_pool_size: int = _env_int("ES_CORE_POOL_SIZE", 4)
    max_pool_size: int = _env_int("ES_MAX_POOL_SIZE", 8)
    http_compress: bool = _env_bool("ES_HTTP_COMPRESS", True)
    health_timeout: int = _env_int("ES_CLUSTER_HEALTH_TIMEOUT_SECONDS", 5)
    feature_index: str = os.getenv("ES_FEATURE_INDEX", "stock-engineered-features")
    prediction_index: str = os.getenv("ES_PREDICTION_INDEX", "stock-predictions")
    dlq_enabled: bool = _env_bool("ES_DLQ_ENABLED", True)
    dlq_path: str = os.getenv("ES_DLQ_PATH", "logs/elasticsearch_deadletter.jsonl")
    circuit_breaker_failure_threshold: int = _env_int("ES_CIRCUIT_BREAKER_FAILURE_THRESHOLD", 3)
    circuit_breaker_reset_seconds: int = _env_int("ES_CIRCUIT_BREAKER_RESET_SECONDS", 30)
    ilm_enabled: bool = _env_bool("ES_ILM_ENABLED", True)
    ilm_policy_name: str = os.getenv("ES_ILM_POLICY_NAME", "stock-analysis-90d-policy")
    index_retention_days: int = _env_int("ES_INDEX_RETENTION_DAYS", 90)
    daily_index_enabled: bool = _env_bool("ES_DAILY_INDEX_ENABLED", True)

    def __post_init__(self) -> None:
        normalized_scheme = self.scheme.strip().lower()
        if normalized_scheme not in {"http", "https"}:
            raise ValueError("ES_SCHEME must be either 'http' or 'https'.")
        object.__setattr__(self, "scheme", normalized_scheme)

        for field_name in (
            "port",
            "request_timeout",
            "batch_size",
            "flush_interval_seconds",
            "core_pool_size",
            "max_pool_size",
            "health_timeout",
            "circuit_breaker_failure_threshold",
            "circuit_breaker_reset_seconds",
            "index_retention_days",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be > 0.")

    @property
    def hosts(self) -> list[str]:
        return [f"{self.scheme}://{self.host}:{self.port}"]


def create_elasticsearch_client(
    config: ElasticsearchSinkConfig | None = None,
    client_cls=None,
) -> Elasticsearch:
    config = config or ElasticsearchSinkConfig()
    if client_cls is None:
        from elasticsearch import Elasticsearch

        client_cls = Elasticsearch

    basic_auth = None
    if config.username and config.password:
        basic_auth = (config.username, config.password)

    return client_cls(
        hosts=config.hosts,
        basic_auth=basic_auth,
        request_timeout=config.request_timeout,
        connections_per_node=config.max_pool_size,
        http_compress=config.http_compress,
        verify_certs=False,
    )


def load_index_mapping(document_type: str) -> dict[str, Any]:
    mapping_path = INDEX_MAPPING_FILES[document_type]
    with mapping_path.open("r", encoding="utf-8") as mapping_file:
        return json.load(mapping_file)


def index_exists(client: Elasticsearch, index_name: str) -> bool:
    return bool(client.indices.exists(index=index_name))


def create_index_if_missing(
    client: Elasticsearch,
    index_name: str,
    mapping: dict[str, Any],
    required_fields: set[str],
) -> bool:
    if index_exists(client, index_name):
        validate_index_mapping(client, index_name, required_fields)
        return False

    client.indices.create(index=index_name, body=mapping)
    return True


def validate_index_mapping(
    client: Elasticsearch,
    index_name: str,
    required_fields: set[str],
) -> bool:
    response = client.indices.get_mapping(index=index_name)
    properties = _mapping_properties(response, index_name)
    missing_fields = sorted(required_fields - set(properties))
    if missing_fields:
        raise ValueError(f"Index {index_name} mapping missing required fields: {', '.join(missing_fields)}")
    return True


def ensure_elasticsearch_indices(
    client: Elasticsearch,
    config: ElasticsearchSinkConfig | None = None,
) -> dict[str, bool]:
    config = config or ElasticsearchSinkConfig()
    return {
        config.feature_index: create_index_if_missing(
            client,
            config.feature_index,
            load_index_mapping(FEATURE_DOCUMENT_TYPE),
            REQUIRED_FEATURE_MAPPING_FIELDS,
        ),
        config.prediction_index: create_index_if_missing(
            client,
            config.prediction_index,
            load_index_mapping(PREDICTION_DOCUMENT_TYPE),
            REQUIRED_PREDICTION_MAPPING_FIELDS,
        ),
    }


def build_daily_index_name(base_index: str, record: dict[str, Any]) -> str:
    timestamp = normalize_timestamp(record)
    index_date = _date_suffix_from_timestamp(timestamp)
    return f"{base_index}-{index_date}"


def resolve_index_name(
    base_index: str,
    record: dict[str, Any],
    config: ElasticsearchSinkConfig | None = None,
) -> str:
    if config is None or not config.daily_index_enabled:
        return base_index
    return build_daily_index_name(base_index, record)


def build_ilm_policy_body(config: ElasticsearchSinkConfig | None = None) -> dict[str, Any]:
    config = config or ElasticsearchSinkConfig()
    return {
        "policy": {
            "phases": {
                "hot": {
                    "actions": {
                        "set_priority": {"priority": 100},
                    }
                },
                "cold": {
                    "min_age": "30d",
                    "actions": {
                        "set_priority": {"priority": 0},
                    },
                },
                "delete": {
                    "min_age": f"{config.index_retention_days}d",
                    "actions": {"delete": {}},
                },
            }
        }
    }


def build_index_template_body(
    document_type: str,
    index_pattern: str,
    config: ElasticsearchSinkConfig | None = None,
) -> dict[str, Any]:
    config = config or ElasticsearchSinkConfig()
    mapping = load_index_mapping(document_type)
    settings = dict(mapping.get("settings", {}))
    settings["index.lifecycle.name"] = config.ilm_policy_name
    return {
        "index_patterns": [index_pattern],
        "template": {
            "settings": settings,
            "mappings": mapping.get("mappings", {}),
        },
    }


def setup_elasticsearch_lifecycle(
    client: Elasticsearch,
    config: ElasticsearchSinkConfig | None = None,
) -> dict[str, Any]:
    config = config or ElasticsearchSinkConfig()
    if not config.ilm_enabled:
        return {"ilm_enabled": False, "policy": None, "templates": []}

    policy_body = build_ilm_policy_body(config)
    client.ilm.put_lifecycle(name=config.ilm_policy_name, policy=policy_body["policy"])

    templates = [
        (
            f"{config.feature_index}-template",
            FEATURE_DOCUMENT_TYPE,
            f"{config.feature_index}-*",
        ),
        (
            f"{config.prediction_index}-template",
            PREDICTION_DOCUMENT_TYPE,
            f"{config.prediction_index}-*",
        ),
    ]
    created_templates: list[str] = []
    for template_name, document_type, index_pattern in templates:
        client.indices.put_index_template(
            name=template_name,
            body=build_index_template_body(document_type, index_pattern, config=config),
        )
        created_templates.append(template_name)

    return {"ilm_enabled": True, "policy": config.ilm_policy_name, "templates": created_templates}


class _NoopCounter:
    def inc(self, value: int | float = 1) -> None:
        return None


class ElasticsearchSinkMetrics:
    def __init__(self, metric_group: Any | None = None) -> None:
        self.documents_inserted = self._counter(metric_group, "documents_inserted")
        self.insert_errors = self._counter(metric_group, "insert_errors")
        self.bulk_flushes = self._counter(metric_group, "bulk_flushes")
        self.bulk_failures = self._counter(metric_group, "bulk_failures")
        self.flush_latency_ms_total = self._counter(metric_group, "bulk_flush_latency_ms_total")
        self.last_flush_latency_ms = 0.0

    def record_success(self, documents_inserted: int, latency_ms: float) -> None:
        self.bulk_flushes.inc()
        if documents_inserted > 0:
            self.documents_inserted.inc(documents_inserted)
        self.record_latency(latency_ms)

    def record_insert_errors(self, count: int) -> None:
        if count > 0:
            self.insert_errors.inc(count)

    def record_failure(self, count: int) -> None:
        self.bulk_failures.inc()
        self.record_insert_errors(count)

    def record_latency(self, latency_ms: float) -> None:
        self.last_flush_latency_ms = latency_ms
        self.flush_latency_ms_total.inc(latency_ms)

    @staticmethod
    def _counter(metric_group: Any | None, name: str) -> Any:
        if metric_group is None:
            return _NoopCounter()
        try:
            return metric_group.counter(name)
        except AttributeError:
            return _NoopCounter()


class ElasticsearchCircuitBreaker:
    def __init__(
        self,
        failure_threshold: int,
        reset_seconds: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self.clock = clock
        self.failure_count = 0
        self.opened_at: float | None = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if self.clock() - self.opened_at >= self.reset_seconds:
            return False
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = self.clock()


def write_dead_letter_entries(entries: Iterable[dict[str, Any]], dlq_path: str) -> None:
    path = Path(dlq_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as dlq_file:
        for entry in entries:
            dlq_file.write(json.dumps(entry, default=str, sort_keys=True) + "\n")


def build_dead_letter_entries(
    records: Iterable[dict[str, Any]],
    index_name: str,
    error_reason: Any,
    errors: Iterable[Any] | None = None,
    operation: str = "index",
) -> list[dict[str, Any]]:
    records_list = list(records)
    errors_list = list(errors or [])
    if not errors_list:
        errors_list = [error_reason] * len(records_list)

    entries: list[dict[str, Any]] = []
    for position, error in enumerate(errors_list):
        document_id = _error_document_id(error)
        source = _record_for_document_id(records_list, document_id)
        if not source:
            source = records_list[position] if position < len(records_list) else {}
        entries.append(
            {
                "@timestamp": _utc_timestamp(),
                "index": index_name,
                "document_id": document_id or build_document_id(source),
                "operation": operation,
                "error_reason": _error_reason(error_reason if error is None else error),
                "source": source,
            }
        )
    return entries


def write_dead_letters_for_records(
    records: Iterable[dict[str, Any]],
    index_name: str,
    error_reason: Any,
    config: ElasticsearchSinkConfig,
    errors: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    entries = build_dead_letter_entries(records, index_name, error_reason, errors=errors)
    if config.dlq_enabled and entries:
        write_dead_letter_entries(entries, config.dlq_path)
    return entries


def build_feature_document(record: dict[str, Any]) -> dict[str, Any]:
    document = dict(record)
    document.setdefault("type", FEATURE_DOCUMENT_TYPE)
    document["@timestamp"] = normalize_timestamp(document)
    return document


def build_prediction_document(record: dict[str, Any]) -> dict[str, Any]:
    document = dict(record)
    document.setdefault("type", PREDICTION_DOCUMENT_TYPE)
    document["@timestamp"] = normalize_timestamp(document)
    return document


def route_document_type(record: dict[str, Any]) -> str:
    record_type = str(record.get("type", "")).strip().lower()
    if record_type == PREDICTION_DOCUMENT_TYPE:
        return PREDICTION_DOCUMENT_TYPE
    if record_type == FEATURE_DOCUMENT_TYPE:
        return FEATURE_DOCUMENT_TYPE
    if any(field in record for field in PREDICTION_MARKER_FIELDS):
        return PREDICTION_DOCUMENT_TYPE
    if any(field in record for field in FEATURE_MARKER_FIELDS):
        return FEATURE_DOCUMENT_TYPE
    return FEATURE_DOCUMENT_TYPE


def is_prediction_record(record: dict[str, Any]) -> bool:
    return route_document_type(record) == PREDICTION_DOCUMENT_TYPE


def is_feature_record(record: dict[str, Any]) -> bool:
    return route_document_type(record) == FEATURE_DOCUMENT_TYPE


def normalize_timestamp(record: dict[str, Any]) -> str:
    raw_value = (
        record.get("@timestamp")
        or record.get("date")
        or record.get("prediction_date")
        or record.get("window_end_ms")
        or record.get("event_ts_ms")
    )
    if raw_value is None:
        return _utc_timestamp()

    if isinstance(raw_value, datetime):
        return _to_utc(raw_value).isoformat()

    if isinstance(raw_value, (int, float)):
        return _timestamp_from_epoch(raw_value).isoformat()

    text = str(raw_value).strip()
    if not text:
        return _utc_timestamp()
    if text.isdigit():
        return _timestamp_from_epoch(float(text)).isoformat()

    try:
        return _to_utc(datetime.fromisoformat(text.replace("Z", "+00:00"))).isoformat()
    except ValueError:
        return _utc_timestamp()


def build_document_id(record: dict[str, Any]) -> str | None:
    ticker = record.get("ticker")
    date_value = record.get("date") or record.get("prediction_date")
    if not ticker or not date_value:
        return None
    return f"{ticker}_{date_value}"


def build_bulk_actions(
    records: Iterable[dict[str, Any]],
    index_name: str,
    config: ElasticsearchSinkConfig | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for record in records:
        payload = dict(record)
        action = {
            "_index": resolve_index_name(index_name, payload, config=config),
            "_source": payload,
        }
        document_id = build_document_id(payload)
        if document_id:
            action["_id"] = document_id
        actions.append(action)
    return actions


def bulk_index_documents(
    client: Elasticsearch,
    records: Iterable[dict[str, Any]],
    index_name: str,
    config: ElasticsearchSinkConfig | None = None,
    bulk_fn: Callable[..., tuple[int, list[Any]]] | None = None,
) -> tuple[int, list[Any]]:
    config = config or ElasticsearchSinkConfig()
    actions = build_bulk_actions(records, index_name=index_name, config=config)
    if not actions:
        return 0, []

    if bulk_fn is None:
        from elasticsearch.helpers import bulk

        bulk_fn = bulk

    success_count, errors = bulk_fn(
        client,
        actions,
        chunk_size=config.batch_size,
        request_timeout=config.request_timeout,
        raise_on_error=False,
    )
    if errors:
        logger.warning("Bulk indexing completed with %s errors for index %s", len(errors), index_name)
    return success_count, errors


def bulk_index_documents_with_retry(
    client: Elasticsearch,
    records: Iterable[dict[str, Any]],
    index_name: str,
    config: ElasticsearchSinkConfig | None = None,
    retries: int = 3,
    base_delay_seconds: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    bulk_fn: Callable[..., tuple[int, list[Any]]] | None = None,
) -> tuple[int, list[Any]]:
    if retries <= 0:
        raise ValueError("retries must be > 0")

    records_list = list(records)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return bulk_index_documents(
                client,
                records_list,
                index_name=index_name,
                config=config,
                bulk_fn=bulk_fn,
            )
        except Exception as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            sleep_fn(base_delay_seconds * (2 ** attempt))

    raise last_error  # type: ignore[misc]


class ElasticsearchSinkFunction:
    def __init__(
        self,
        index_name: str,
        config: ElasticsearchSinkConfig | None = None,
        client: Any | None = None,
        document_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        bulk_fn: Callable[..., tuple[int, list[Any]]] | None = None,
        metric_group: Any | None = None,
        metrics: ElasticsearchSinkMetrics | None = None,
        circuit_breaker: ElasticsearchCircuitBreaker | None = None,
        clock: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.index_name = index_name
        self.config = config or ElasticsearchSinkConfig()
        self.client = client
        self.document_builder = document_builder or build_feature_document
        self.bulk_fn = bulk_fn
        self.buffer: list[dict[str, Any]] = []
        self.metrics = metrics or ElasticsearchSinkMetrics(metric_group)
        self.circuit_breaker = circuit_breaker or ElasticsearchCircuitBreaker(
            self.config.circuit_breaker_failure_threshold,
            self.config.circuit_breaker_reset_seconds,
            clock=clock,
        )
        self.clock = clock
        self.sleep_fn = sleep_fn

    def invoke(self, record: dict[str, Any]) -> None:
        self.buffer.append(self.document_builder(record))
        if len(self.buffer) >= self.config.batch_size:
            self.flush()

    def flush(self) -> tuple[int, list[Any]]:
        if not self.buffer:
            return 0, []

        records = list(self.buffer)
        if self.circuit_breaker.is_open():
            reason = "elasticsearch circuit breaker is open"
            write_dead_letters_for_records(records, self.index_name, reason, self.config)
            self.metrics.record_failure(len(records))
            self.buffer.clear()
            raise RuntimeError(reason)

        client = self._client()
        start_time = self.clock()
        try:
            success_count, errors = bulk_index_documents_with_retry(
                client,
                records,
                index_name=self.index_name,
                config=self.config,
                bulk_fn=self.bulk_fn,
                sleep_fn=self.sleep_fn,
            )
        except Exception as exc:
            latency_ms = max((self.clock() - start_time) * 1000.0, 0.0)
            self.metrics.record_latency(latency_ms)
            self.metrics.record_failure(len(records))
            self.circuit_breaker.record_failure()
            write_dead_letters_for_records(records, self.index_name, str(exc), self.config)
            self.buffer.clear()
            raise

        latency_ms = max((self.clock() - start_time) * 1000.0, 0.0)
        self.metrics.record_success(success_count, latency_ms)
        if errors:
            self.metrics.record_insert_errors(len(errors))
            write_dead_letters_for_records(records, self.index_name, "bulk item error", self.config, errors=errors)
            logger.warning("Bulk indexing returned %s item errors for index %s", len(errors), self.index_name)
        self.circuit_breaker.record_success()
        self.buffer.clear()
        return success_count, errors

    def close(self) -> tuple[int, list[Any]]:
        return self.flush()

    def _client(self) -> Any:
        if self.client is None:
            self.client = create_elasticsearch_client(self.config)
        return self.client


class DualIndexElasticsearchSink:
    def __init__(
        self,
        config: ElasticsearchSinkConfig | None = None,
        client: Any | None = None,
        feature_sink: ElasticsearchSinkFunction | None = None,
        prediction_sink: ElasticsearchSinkFunction | None = None,
        bulk_fn: Callable[..., tuple[int, list[Any]]] | None = None,
        metric_group: Any | None = None,
        clock: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or ElasticsearchSinkConfig()
        self.feature_sink = feature_sink or ElasticsearchSinkFunction(
            index_name=self.config.feature_index,
            config=self.config,
            client=client,
            document_builder=build_feature_document,
            bulk_fn=bulk_fn,
            metric_group=metric_group,
            clock=clock,
            sleep_fn=sleep_fn,
        )
        self.prediction_sink = prediction_sink or ElasticsearchSinkFunction(
            index_name=self.config.prediction_index,
            config=self.config,
            client=client,
            document_builder=build_prediction_document,
            bulk_fn=bulk_fn,
            metric_group=metric_group,
            clock=clock,
            sleep_fn=sleep_fn,
        )

    def invoke(self, record: dict[str, Any]) -> None:
        if is_prediction_record(record):
            self.prediction_sink.invoke(record)
        else:
            self.feature_sink.invoke(record)

    def flush(self) -> dict[str, tuple[int, list[Any]]]:
        return {
            FEATURE_DOCUMENT_TYPE: self.feature_sink.flush(),
            PREDICTION_DOCUMENT_TYPE: self.prediction_sink.flush(),
        }

    def close(self) -> dict[str, tuple[int, list[Any]]]:
        return {
            FEATURE_DOCUMENT_TYPE: self.feature_sink.close(),
            PREDICTION_DOCUMENT_TYPE: self.prediction_sink.close(),
        }


def check_cluster_health(
    client: Elasticsearch,
    config: ElasticsearchSinkConfig | None = None,
) -> dict[str, Any]:
    config = config or ElasticsearchSinkConfig()
    health = client.cluster.health(timeout=f"{config.health_timeout}s")
    logger.info(
        "Elasticsearch cluster health=%s active_shards=%s",
        health.get("status"),
        health.get("active_shards"),
    )
    return health


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timestamp_from_epoch(value: int | float) -> datetime:
    timestamp = float(value)
    if timestamp > 10_000_000_000:
        timestamp = timestamp / 1000.0
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def _date_suffix_from_timestamp(timestamp: str) -> str:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return _to_utc(parsed).date().isoformat()


def _mapping_properties(response: dict[str, Any], index_name: str) -> dict[str, Any]:
    if "mappings" in response:
        return response.get("mappings", {}).get("properties", {})
    return response.get(index_name, {}).get("mappings", {}).get("properties", {})


def _error_document_id(error: Any) -> str | None:
    if not isinstance(error, dict):
        return None
    payload = _error_payload(error)
    document_id = payload.get("_id")
    return str(document_id) if document_id else None


def _error_reason(error: Any) -> str:
    if isinstance(error, Exception):
        return str(error)
    if not isinstance(error, dict):
        return str(error)

    payload = _error_payload(error)
    reason = payload.get("error", error)
    if isinstance(reason, dict):
        return reason.get("reason") or json.dumps(reason, default=str, sort_keys=True)
    return str(reason)


def _error_payload(error: dict[str, Any]) -> dict[str, Any]:
    if len(error) == 1:
        value = next(iter(error.values()))
        if isinstance(value, dict):
            return value
    return error


def _record_for_document_id(records: Iterable[dict[str, Any]], document_id: str | None) -> dict[str, Any]:
    if not document_id:
        return {}
    for record in records:
        if build_document_id(record) == document_id:
            return record
    return {}
