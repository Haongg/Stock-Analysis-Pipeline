import json
import tempfile
from unittest.mock import Mock

from src.flink.es_sink import (
    DualIndexElasticsearchSink,
    ElasticsearchCircuitBreaker,
    ElasticsearchSinkFunction,
    ElasticsearchSinkConfig,
    ElasticsearchSinkMetrics,
    build_daily_index_name,
    build_dead_letter_entries,
    build_bulk_actions,
    build_document_id,
    build_feature_document,
    build_ilm_policy_body,
    build_index_template_body,
    build_prediction_document,
    bulk_index_documents_with_retry,
    check_cluster_health,
    create_index_if_missing,
    create_elasticsearch_client,
    ensure_elasticsearch_indices,
    is_feature_record,
    is_prediction_record,
    load_index_mapping,
    normalize_timestamp,
    resolve_index_name,
    route_document_type,
    setup_elasticsearch_lifecycle,
    validate_index_mapping,
    write_dead_letter_entries,
)


class _Counter:
    def __init__(self) -> None:
        self.value = 0

    def inc(self, amount=1) -> None:
        self.value += amount


class _MetricGroup:
    def __init__(self) -> None:
        self.counters: dict[str, _Counter] = {}

    def counter(self, name: str) -> _Counter:
        self.counters.setdefault(name, _Counter())
        return self.counters[name]


def test_build_feature_document_sets_defaults() -> None:
    document = build_feature_document({"ticker": "AAPL", "date": "2025-01-15"})
    assert document["type"] == "features"
    assert document["ticker"] == "AAPL"
    assert "@timestamp" in document


def test_build_prediction_document_sets_defaults() -> None:
    document = build_prediction_document({"ticker": "AAPL", "prediction_date": "2025-01-16"})
    assert document["type"] == "prediction"
    assert document["ticker"] == "AAPL"
    assert "@timestamp" in document


def test_document_routing_uses_explicit_type_and_prediction_markers() -> None:
    assert route_document_type({"type": "prediction", "ticker": "AAPL"}) == "prediction"
    assert route_document_type({"type": "features", "ticker": "AAPL"}) == "features"
    assert route_document_type({"ticker": "AAPL", "predicted_close": 150.75}) == "prediction"
    assert route_document_type({"ticker": "AAPL", "model_version": "1"}) == "prediction"
    assert route_document_type({"ticker": "AAPL", "date": "2025-01-15", "close": 150.0}) == "features"
    assert route_document_type({"ticker": "AAPL"}) == "features"


def test_document_routing_predicates_match_route() -> None:
    assert is_prediction_record({"ticker": "AAPL", "prediction_error": "model_failed"}) is True
    assert is_feature_record({"ticker": "AAPL", "date": "2025-01-15", "close": 150.0}) is True
    assert is_feature_record({"ticker": "AAPL", "prediction_date": "2025-01-16"}) is False


def test_normalize_timestamp_from_iso_millis_and_date_fields() -> None:
    assert normalize_timestamp({"@timestamp": "2025-01-15T10:30:00Z"}) == "2025-01-15T10:30:00+00:00"
    assert normalize_timestamp({"event_ts_ms": 1736937000000}) == "2025-01-15T10:30:00+00:00"
    assert normalize_timestamp({"date": "2025-01-15"}) == "2025-01-15T00:00:00+00:00"


def test_normalize_timestamp_falls_back_to_current_utc() -> None:
    timestamp = normalize_timestamp({})

    assert "T" in timestamp
    assert timestamp.endswith("+00:00")


def test_build_document_id_uses_ticker_and_date_or_prediction_date() -> None:
    assert build_document_id({"ticker": "AAPL", "date": "2025-01-15"}) == "AAPL_2025-01-15"
    assert build_document_id({"ticker": "MSFT", "prediction_date": "2025-01-16"}) == "MSFT_2025-01-16"
    assert build_document_id({"ticker": "MSFT"}) is None
    assert build_document_id({"date": "2025-01-15"}) is None


def test_build_bulk_actions_uses_stable_document_id() -> None:
    actions = build_bulk_actions(
        [
            {"ticker": "AAPL", "date": "2025-01-15", "close": 150.0},
            {"ticker": "MSFT", "prediction_date": "2025-01-16", "predicted_close": 420.0},
        ],
        index_name="stock-engineered-features",
    )

    assert actions[0]["_id"] == "AAPL_2025-01-15"
    assert actions[1]["_id"] == "MSFT_2025-01-16"
    assert actions[0]["_index"] == "stock-engineered-features"


def test_build_bulk_actions_omits_id_when_not_available() -> None:
    actions = build_bulk_actions([{"ticker": "AAPL", "close": 150.0}], index_name="stock-engineered-features")

    assert "_id" not in actions[0]


def test_daily_index_name_from_feature_and_prediction_dates() -> None:
    assert (
        build_daily_index_name("stock-engineered-features", {"date": "2025-01-15"})
        == "stock-engineered-features-2025-01-15"
    )
    assert (
        build_daily_index_name("stock-predictions", {"prediction_date": "2025-01-16"})
        == "stock-predictions-2025-01-16"
    )


def test_resolve_index_name_respects_daily_index_toggle() -> None:
    record = {"ticker": "AAPL", "date": "2025-01-15"}

    assert (
        resolve_index_name("stock-engineered-features", record, ElasticsearchSinkConfig(daily_index_enabled=True))
        == "stock-engineered-features-2025-01-15"
    )
    assert (
        resolve_index_name("stock-engineered-features", record, ElasticsearchSinkConfig(daily_index_enabled=False))
        == "stock-engineered-features"
    )


def test_bulk_actions_use_daily_indices_when_configured() -> None:
    actions = build_bulk_actions(
        [
            {"ticker": "AAPL", "date": "2025-01-15", "close": 150.0},
            {"ticker": "MSFT", "date": "2025-01-16", "close": 420.0},
        ],
        index_name="stock-engineered-features",
        config=ElasticsearchSinkConfig(daily_index_enabled=True),
    )

    assert actions[0]["_index"] == "stock-engineered-features-2025-01-15"
    assert actions[1]["_index"] == "stock-engineered-features-2025-01-16"


def test_config_builds_canonical_host_url_and_default_indices() -> None:
    config = ElasticsearchSinkConfig(host="es01", port=9200, scheme="https")

    assert config.hosts == ["https://es01:9200"]
    assert config.batch_size == 100
    assert config.flush_interval_seconds == 5
    assert config.feature_index == "stock-engineered-features"
    assert config.prediction_index == "stock-predictions"
    assert config.ilm_enabled is True
    assert config.ilm_policy_name == "stock-analysis-90d-policy"
    assert config.index_retention_days == 90
    assert config.daily_index_enabled is True


def test_mapping_files_use_expected_index_settings() -> None:
    feature_mapping = load_index_mapping("features")
    prediction_mapping = load_index_mapping("prediction")

    for mapping in (feature_mapping, prediction_mapping):
        assert mapping["settings"]["number_of_shards"] == 3
        assert mapping["settings"]["number_of_replicas"] == 1
        assert mapping["settings"]["refresh_interval"] == "30s"


def test_create_index_if_missing_creates_with_mapping_body() -> None:
    client = Mock()
    client.indices.exists.return_value = False
    mapping = load_index_mapping("features")

    created = create_index_if_missing(
        client,
        "stock-engineered-features",
        mapping,
        {"@timestamp", "ticker"},
    )

    assert created is True
    client.indices.create.assert_called_once_with(index="stock-engineered-features", body=mapping)
    client.indices.get_mapping.assert_not_called()


def test_create_index_if_missing_validates_existing_index() -> None:
    client = Mock()
    client.indices.exists.return_value = True
    client.indices.get_mapping.return_value = {
        "stock-engineered-features": {
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "ticker": {"type": "keyword"},
                }
            }
        }
    }

    created = create_index_if_missing(
        client,
        "stock-engineered-features",
        load_index_mapping("features"),
        {"@timestamp", "ticker"},
    )

    assert created is False
    client.indices.create.assert_not_called()
    client.indices.get_mapping.assert_called_once_with(index="stock-engineered-features")


def test_validate_index_mapping_raises_when_required_fields_missing() -> None:
    client = Mock()
    client.indices.get_mapping.return_value = {
        "stock-predictions": {
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "ticker": {"type": "keyword"},
                }
            }
        }
    }

    try:
        validate_index_mapping(client, "stock-predictions", {"@timestamp", "ticker", "predicted_close"})
    except ValueError as exc:
        assert "predicted_close" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing mapping field")


def test_ensure_elasticsearch_indices_creates_feature_and_prediction_indices() -> None:
    client = Mock()
    client.indices.exists.side_effect = [False, False]
    config = ElasticsearchSinkConfig(
        feature_index="custom-features",
        prediction_index="custom-predictions",
    )

    results = ensure_elasticsearch_indices(client, config=config)

    assert results == {"custom-features": True, "custom-predictions": True}
    created_indices = [call.kwargs["index"] for call in client.indices.create.call_args_list]
    assert created_indices == ["custom-features", "custom-predictions"]


def test_ensure_elasticsearch_indices_skips_existing_valid_indices() -> None:
    client = Mock()
    client.indices.exists.side_effect = [True, True]
    client.indices.get_mapping.side_effect = [
        {"stock-engineered-features": {"mappings": load_index_mapping("features")["mappings"]}},
        {"stock-predictions": {"mappings": load_index_mapping("prediction")["mappings"]}},
    ]

    results = ensure_elasticsearch_indices(client)

    assert results == {"stock-engineered-features": False, "stock-predictions": False}
    client.indices.create.assert_not_called()


def test_ilm_policy_body_uses_configured_retention_days() -> None:
    policy = build_ilm_policy_body(ElasticsearchSinkConfig(index_retention_days=120))

    assert policy["policy"]["phases"]["delete"]["min_age"] == "120d"
    assert "cold" in policy["policy"]["phases"]


def test_index_template_body_attaches_lifecycle_policy_and_pattern() -> None:
    template = build_index_template_body(
        "features",
        "stock-engineered-features-*",
        config=ElasticsearchSinkConfig(ilm_policy_name="custom-policy"),
    )

    assert template["index_patterns"] == ["stock-engineered-features-*"]
    assert template["template"]["settings"]["index.lifecycle.name"] == "custom-policy"
    assert "mappings" in template["template"]


def test_setup_elasticsearch_lifecycle_puts_policy_and_templates() -> None:
    client = Mock()
    config = ElasticsearchSinkConfig(ilm_policy_name="stock-policy")

    result = setup_elasticsearch_lifecycle(client, config=config)

    assert result == {
        "ilm_enabled": True,
        "policy": "stock-policy",
        "templates": ["stock-engineered-features-template", "stock-predictions-template"],
    }
    client.ilm.put_lifecycle.assert_called_once()
    assert client.ilm.put_lifecycle.call_args.kwargs["name"] == "stock-policy"
    template_names = [call.kwargs["name"] for call in client.indices.put_index_template.call_args_list]
    assert template_names == ["stock-engineered-features-template", "stock-predictions-template"]


def test_setup_elasticsearch_lifecycle_skips_when_disabled() -> None:
    client = Mock()

    result = setup_elasticsearch_lifecycle(client, config=ElasticsearchSinkConfig(ilm_enabled=False))

    assert result == {"ilm_enabled": False, "policy": None, "templates": []}
    client.ilm.put_lifecycle.assert_not_called()
    client.indices.put_index_template.assert_not_called()


def test_create_client_omits_auth_when_credentials_missing() -> None:
    client_cls = Mock(return_value="client")
    config = ElasticsearchSinkConfig(username="elastic", password=None)

    client = create_elasticsearch_client(config=config, client_cls=client_cls)

    assert client == "client"
    kwargs = client_cls.call_args.kwargs
    assert kwargs["hosts"] == ["http://localhost:9200"]
    assert kwargs["basic_auth"] is None
    assert kwargs["http_compress"] is True
    assert kwargs["connections_per_node"] == 8


def test_create_client_uses_configured_pool_and_compression_settings() -> None:
    client_cls = Mock(return_value="client")
    config = ElasticsearchSinkConfig(max_pool_size=12, http_compress=False)

    create_elasticsearch_client(config=config, client_cls=client_cls)

    kwargs = client_cls.call_args.kwargs
    assert kwargs["connections_per_node"] == 12
    assert kwargs["http_compress"] is False


def test_create_client_uses_basic_auth_when_credentials_present() -> None:
    client_cls = Mock(return_value="client")
    config = ElasticsearchSinkConfig(username="elastic", password="changeme")

    create_elasticsearch_client(config=config, client_cls=client_cls)

    assert client_cls.call_args.kwargs["basic_auth"] == ("elastic", "changeme")


def test_bulk_index_with_retry_succeeds_after_transient_failure() -> None:
    calls = {"count": 0}
    delays: list[float] = []

    def flaky_bulk(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary es failure")
        return 1, []

    result = bulk_index_documents_with_retry(
        client=Mock(),
        records=[{"ticker": "AAPL", "date": "2025-01-15"}],
        index_name="stock-engineered-features",
        config=ElasticsearchSinkConfig(),
        sleep_fn=delays.append,
        bulk_fn=flaky_bulk,
    )

    assert result == (1, [])
    assert calls["count"] == 2
    assert delays == [1.0]


def test_bulk_index_uses_configured_chunk_size_and_request_timeout() -> None:
    captured: dict = {}

    def fake_bulk(client, actions, **kwargs):
        captured["actions"] = actions
        captured.update(kwargs)
        return len(actions), []

    result = bulk_index_documents_with_retry(
        client=Mock(),
        records=[
            {"ticker": "AAPL", "date": "2025-01-15"},
            {"ticker": "MSFT", "date": "2025-01-15"},
        ],
        index_name="stock-engineered-features",
        config=ElasticsearchSinkConfig(batch_size=50, request_timeout=17),
        bulk_fn=fake_bulk,
    )

    assert result == (2, [])
    assert captured["chunk_size"] == 50
    assert captured["request_timeout"] == 17
    assert captured["raise_on_error"] is False
    assert len(captured["actions"]) == 2


def test_bulk_index_with_retry_raises_after_exhausted_attempts() -> None:
    delays: list[float] = []

    def failing_bulk(*args, **kwargs):
        raise RuntimeError("still down")

    try:
        bulk_index_documents_with_retry(
            client=Mock(),
            records=[{"ticker": "AAPL", "date": "2025-01-15"}],
            index_name="stock-engineered-features",
            config=ElasticsearchSinkConfig(),
            sleep_fn=delays.append,
            bulk_fn=failing_bulk,
        )
    except RuntimeError as exc:
        assert str(exc) == "still down"
    else:
        raise AssertionError("Expected RuntimeError after retry exhaustion")

    assert delays == [1.0, 2.0]


def test_sink_buffers_until_batch_size_then_flushes() -> None:
    calls: list[list[dict]] = []

    def fake_bulk(client, actions, **kwargs):
        calls.append(actions)
        return len(actions), []

    sink = ElasticsearchSinkFunction(
        index_name="stock-engineered-features",
        config=ElasticsearchSinkConfig(batch_size=2),
        client=Mock(),
        bulk_fn=fake_bulk,
    )

    sink.invoke({"ticker": "AAPL", "date": "2025-01-15"})
    assert len(sink.buffer) == 1
    assert calls == []

    sink.invoke({"ticker": "MSFT", "date": "2025-01-15"})
    assert sink.buffer == []
    assert len(calls) == 1
    assert len(calls[0]) == 2


def test_sink_close_flushes_remaining_records() -> None:
    calls: list[list[dict]] = []

    def fake_bulk(client, actions, **kwargs):
        calls.append(actions)
        return len(actions), []

    sink = ElasticsearchSinkFunction(
        index_name="stock-engineered-features",
        config=ElasticsearchSinkConfig(batch_size=10),
        client=Mock(),
        bulk_fn=fake_bulk,
    )
    sink.invoke({"ticker": "AAPL", "date": "2025-01-15"})

    result = sink.close()

    assert result == (1, [])
    assert sink.buffer == []
    assert len(calls) == 1


def test_sink_successful_flush_records_metrics() -> None:
    metric_group = _MetricGroup()

    def fake_bulk(client, actions, **kwargs):
        return len(actions), []

    sink = ElasticsearchSinkFunction(
        index_name="stock-engineered-features",
        config=ElasticsearchSinkConfig(batch_size=2),
        client=Mock(),
        bulk_fn=fake_bulk,
        metric_group=metric_group,
        clock=lambda: 10.0,
    )

    sink.invoke({"ticker": "AAPL", "date": "2025-01-15"})
    sink.invoke({"ticker": "MSFT", "date": "2025-01-15"})

    assert metric_group.counters["documents_inserted"].value == 2
    assert metric_group.counters["bulk_flushes"].value == 1
    assert metric_group.counters["insert_errors"].value == 0
    assert sink.metrics.last_flush_latency_ms == 0.0


def test_bulk_item_errors_are_written_to_dead_letter_queue() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dlq_path = f"{tmpdir}/es_dlq.jsonl"
        metric_group = _MetricGroup()

        def fake_bulk(client, actions, **kwargs):
            return 1, [
                {
                    "index": {
                        "_id": "MSFT_2025-01-15",
                        "error": {"type": "mapper_parsing_exception", "reason": "bad volume"},
                    }
                }
            ]

        sink = ElasticsearchSinkFunction(
            index_name="stock-engineered-features",
            config=ElasticsearchSinkConfig(batch_size=2, dlq_path=dlq_path),
            client=Mock(),
            bulk_fn=fake_bulk,
            metric_group=metric_group,
        )

        sink.invoke({"ticker": "AAPL", "date": "2025-01-15", "close": 150.0})
        sink.invoke({"ticker": "MSFT", "date": "2025-01-15", "close": 420.0})

        with open(dlq_path, "r", encoding="utf-8") as dlq_file:
            entries = [json.loads(line) for line in dlq_file]

        assert metric_group.counters["documents_inserted"].value == 1
        assert metric_group.counters["insert_errors"].value == 1
        assert len(entries) == 1
        assert entries[0]["index"] == "stock-engineered-features"
        assert entries[0]["document_id"] == "MSFT_2025-01-15"
        assert entries[0]["error_reason"] == "bad volume"
        assert entries[0]["operation"] == "index"
        assert "@timestamp" in entries[0]


def test_bulk_exception_writes_buffer_to_dead_letter_queue_and_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dlq_path = f"{tmpdir}/es_dlq.jsonl"
        metric_group = _MetricGroup()

        def failing_bulk(client, actions, **kwargs):
            raise RuntimeError("es down")

        sink = ElasticsearchSinkFunction(
            index_name="stock-engineered-features",
            config=ElasticsearchSinkConfig(batch_size=1, dlq_path=dlq_path),
            client=Mock(),
            bulk_fn=failing_bulk,
            metric_group=metric_group,
            sleep_fn=lambda delay: None,
        )

        try:
            sink.invoke({"ticker": "AAPL", "date": "2025-01-15", "close": 150.0})
        except RuntimeError as exc:
            assert str(exc) == "es down"
        else:
            raise AssertionError("Expected RuntimeError for failed flush")

        with open(dlq_path, "r", encoding="utf-8") as dlq_file:
            entries = [json.loads(line) for line in dlq_file]

        assert metric_group.counters["bulk_failures"].value == 1
        assert metric_group.counters["insert_errors"].value == 1
        assert sink.buffer == []
        assert len(entries) == 1
        assert entries[0]["error_reason"] == "es down"
        assert entries[0]["source"]["ticker"] == "AAPL"


def test_circuit_breaker_opens_and_fails_fast_until_reset_window() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dlq_path = f"{tmpdir}/es_dlq.jsonl"
        now = {"value": 0.0}
        calls = {"count": 0}

        def clock() -> float:
            return now["value"]

        def flaky_bulk(client, actions, **kwargs):
            calls["count"] += 1
            if calls["count"] <= 3:
                raise RuntimeError("temporary outage")
            return len(actions), []

        sink = ElasticsearchSinkFunction(
            index_name="stock-engineered-features",
            config=ElasticsearchSinkConfig(
                batch_size=1,
                dlq_path=dlq_path,
                circuit_breaker_failure_threshold=1,
                circuit_breaker_reset_seconds=30,
            ),
            client=Mock(),
            bulk_fn=flaky_bulk,
            clock=clock,
            sleep_fn=lambda delay: None,
        )

        try:
            sink.invoke({"ticker": "AAPL", "date": "2025-01-15", "close": 150.0})
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected outage failure")

        assert sink.circuit_breaker.is_open() is True
        assert calls["count"] == 3

        try:
            sink.invoke({"ticker": "GOOG", "date": "2025-01-15", "close": 140.0})
        except RuntimeError as exc:
            assert "circuit breaker is open" in str(exc)
        else:
            raise AssertionError("Expected open circuit failure")

        assert calls["count"] == 3

        now["value"] = 31.0
        sink.invoke({"ticker": "NVDA", "date": "2025-01-15", "close": 900.0})

        assert calls["count"] == 4
        assert sink.circuit_breaker.is_open() is False


def test_dead_letter_helpers_write_valid_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dlq_path = f"{tmpdir}/es_dlq.jsonl"
        entries = build_dead_letter_entries(
            [{"ticker": "AAPL", "date": "2025-01-15"}],
            index_name="stock-engineered-features",
            error_reason="failed insert",
        )

        write_dead_letter_entries(entries, dlq_path)

        with open(dlq_path, "r", encoding="utf-8") as dlq_file:
            written = json.loads(dlq_file.readline())

        assert written["index"] == "stock-engineered-features"
        assert written["document_id"] == "AAPL_2025-01-15"
        assert written["error_reason"] == "failed insert"
        assert written["source"]["ticker"] == "AAPL"


def test_dual_index_sink_routes_features_and_predictions_to_configured_indices() -> None:
    calls: list[list[dict]] = []

    def fake_bulk(client, actions, **kwargs):
        calls.append(actions)
        return len(actions), []

    config = ElasticsearchSinkConfig(
        batch_size=2,
        feature_index="custom-features",
        prediction_index="custom-predictions",
    )
    sink = DualIndexElasticsearchSink(config=config, client=Mock(), bulk_fn=fake_bulk)

    sink.invoke({"ticker": "AAPL", "date": "2025-01-15", "close": 150.0})
    sink.invoke({"ticker": "AAPL", "prediction_date": "2025-01-16", "predicted_close": 151.0})
    assert calls == []

    result = sink.flush()

    assert result["features"] == (1, [])
    assert result["prediction"] == (1, [])
    assert len(calls) == 2
    assert calls[0][0]["_index"] == "custom-features-2025-01-15"
    assert calls[0][0]["_source"]["type"] == "features"
    assert "@timestamp" in calls[0][0]["_source"]
    assert calls[1][0]["_index"] == "custom-predictions-2025-01-16"
    assert calls[1][0]["_source"]["type"] == "prediction"
    assert "@timestamp" in calls[1][0]["_source"]


def test_dual_index_sink_flushes_buffers_independently_by_batch_size() -> None:
    calls: list[list[dict]] = []

    def fake_bulk(client, actions, **kwargs):
        calls.append(actions)
        return len(actions), []

    sink = DualIndexElasticsearchSink(
        config=ElasticsearchSinkConfig(batch_size=2),
        client=Mock(),
        bulk_fn=fake_bulk,
    )

    sink.invoke({"ticker": "AAPL", "date": "2025-01-15", "close": 150.0})
    sink.invoke({"ticker": "MSFT", "date": "2025-01-15", "close": 420.0})

    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert calls[0][0]["_index"] == "stock-engineered-features-2025-01-15"
    assert sink.feature_sink.buffer == []
    assert sink.prediction_sink.buffer == []

    sink.invoke({"ticker": "AAPL", "prediction_date": "2025-01-16", "predicted_close": 151.0})

    assert len(calls) == 1
    assert len(sink.prediction_sink.buffer) == 1


def test_dual_index_sink_close_flushes_both_sinks() -> None:
    calls: list[list[dict]] = []

    def fake_bulk(client, actions, **kwargs):
        calls.append(actions)
        return len(actions), []

    sink = DualIndexElasticsearchSink(
        config=ElasticsearchSinkConfig(batch_size=10),
        client=Mock(),
        bulk_fn=fake_bulk,
    )
    sink.invoke({"ticker": "AAPL", "date": "2025-01-15", "close": 150.0})
    sink.invoke({"ticker": "AAPL", "prediction_date": "2025-01-16", "predicted_close": 151.0})

    result = sink.close()

    assert result == {"features": (1, []), "prediction": (1, [])}
    assert sink.feature_sink.buffer == []
    assert sink.prediction_sink.buffer == []
    assert len(calls) == 2


def test_invalid_config_values_raise_value_error() -> None:
    invalid_configs = [
        {"scheme": "ftp"},
        {"port": 0},
        {"batch_size": 0},
        {"flush_interval_seconds": 0},
        {"core_pool_size": 0},
        {"max_pool_size": 0},
        {"health_timeout": 0},
        {"index_retention_days": 0},
    ]

    for kwargs in invalid_configs:
        try:
            ElasticsearchSinkConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")


def test_check_cluster_health_uses_configured_timeout() -> None:
    client = Mock()
    client.cluster.health.return_value = {"status": "green", "active_shards": 3}
    config = ElasticsearchSinkConfig(health_timeout=9)

    health = check_cluster_health(client, config=config)

    client.cluster.health.assert_called_once_with(timeout="9s")
    assert health["status"] == "green"
