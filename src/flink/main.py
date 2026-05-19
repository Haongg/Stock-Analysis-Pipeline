import os
import logging
import pickle

from pyflink.common import Configuration
from pyflink.common import Duration
from pyflink.common.watermark_strategy import WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.common.time import Time as FlinkTime
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import (
    RichMapFunction,
    RuntimeContext,
    ProcessWindowFunction,
    KeyedProcessFunction,
    AggregateFunction,
)
from pyflink.datastream.connectors.kafka import KafkaSource
from pyflink.datastream.state import (
    ValueStateDescriptor,
    MapStateDescriptor,
    StateTtlConfig,
)
from pyflink.datastream.window import SlidingEventTimeWindows, TumblingEventTimeWindows, Time

from src.flink.monitoring import (
    sequence_metric_classification,
)
from src.flink.indicator_enrichment import build_indicator_history, enrich_record_with_indicators
from src.flink.inference_integration import LSTMInferenceMapFunction
from src.flink.parsing import parse_ohlcv_event
from src.flink.sequence_buffer import append_feature_vector_to_buffer, with_sequence_metadata


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def build_env() -> StreamExecutionEnvironment:
    """
    Create and configure Flink StreamExecutionEnvironment.
    Task 1.1 scope: base environment, parallelism and checkpoint settings.
    """
    config = Configuration()
    config.set_string("pipeline.name", "stock-analysis-pipeline")
    if _env_str("FLINK_ENABLE_ROCKSDB_STATE_BACKEND", "true").lower() == "true":
        config.set_string("state.backend", "rocksdb")
        config.set_string(
            "state.checkpoints.dir",
            _env_str("FLINK_CHECKPOINTS_DIR", "file:///tmp/flink-checkpoints"),
        )
        config.set_string("state.backend.incremental", "true")

    env = StreamExecutionEnvironment.get_execution_environment(config)
    env.set_parallelism(_env_int("FLINK_PARALLELISM", 2))
    env.enable_checkpointing(_env_int("FLINK_CHECKPOINT_INTERVAL_MS", 60000))
    return env


def build_kafka_source() -> KafkaSource:
    """
    Build Kafka source config for topic stock.raw.ohlcv.
    Task 1.2 will replace schema with stricter JSON -> dict parsing logic.
    """
    bootstrap_servers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        "kafka-1:9092,kafka-2:9092,kafka-3:9092",
    )
    topic = os.getenv("KAFKA_TOPIC_RAW", "stock.raw.ohlcv")
    group_id = os.getenv("FLINK_KAFKA_GROUP_ID", "flink-stock-analysis")

    return (
        KafkaSource.builder()
        .set_bootstrap_servers(bootstrap_servers)
        .set_topics(topic)
        .set_group_id(group_id)
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


class ParseOHLCVEventMap(RichMapFunction):
    """Parse + validate Kafka event with counters and warning logs."""

    def open(self, runtime_context: RuntimeContext) -> None:
        metric_group = runtime_context.get_metric_group()
        self.records_total = metric_group.counter("records_total")
        self.records_valid = metric_group.counter("records_valid")
        self.records_invalid_json = metric_group.counter("records_invalid_json")
        self.records_invalid_schema = metric_group.counter("records_invalid_schema")
        self.records_invalid_timestamp = metric_group.counter("records_invalid_timestamp")

    def map(self, value: str) -> dict | None:
        self.records_total.inc()
        parsed = parse_ohlcv_event(value)
        if parsed is not None:
            self.records_valid.inc()
            return parsed

        self._count_invalid_reason(value)
        return None

    def _count_invalid_reason(self, value: str) -> None:
        if not isinstance(value, str):
            self.records_invalid_json.inc()
            logger.warning("Invalid event: non-string payload type")
            return

        stripped = value.lstrip()
        if not stripped or stripped[0] not in "{[":
            self.records_invalid_json.inc()
            logger.warning("Invalid JSON payload: %s", value[:200])
            return

        if '"date"' in value:
            self.records_invalid_timestamp.inc()
            logger.warning("Invalid event timestamp or parse error: %s", value[:200])
            return

        self.records_invalid_schema.inc()
        logger.warning("Invalid event schema: %s", value[:200])


def _init_ohlcv_accumulator() -> dict:
    return {
        "first_ts": None,
        "open": None,
        "high": None,
        "low": None,
        "last_ts": None,
        "close": None,
        "volume_sum": 0.0,
        "event_count": 0,
        "min_price": None,
        "max_price": None,
        "last_event_ts_ms": None,
    }


def _accumulate_ohlcv_event(acc: dict, event: dict) -> dict:
    ts = int(event["event_ts_ms"])
    open_price = float(event["open"])
    high_price = float(event["high"])
    low_price = float(event["low"])
    close_price = float(event["close"])
    volume = float(event["volume"])

    if acc["first_ts"] is None or ts < acc["first_ts"]:
        acc["first_ts"] = ts
        acc["open"] = open_price

    if (
        acc["last_ts"] is None
        or ts > acc["last_ts"]
        or (ts == acc["last_ts"] and acc["close"] is not None)
    ):
        acc["last_ts"] = ts
        acc["close"] = close_price

    acc["high"] = high_price if acc["high"] is None else max(acc["high"], high_price)
    acc["low"] = low_price if acc["low"] is None else min(acc["low"], low_price)
    acc["volume_sum"] += volume
    acc["event_count"] += 1
    acc["min_price"] = low_price if acc["min_price"] is None else min(acc["min_price"], low_price)
    acc["max_price"] = high_price if acc["max_price"] is None else max(acc["max_price"], high_price)
    acc["last_event_ts_ms"] = ts if acc["last_event_ts_ms"] is None else max(acc["last_event_ts_ms"], ts)
    return acc


def _merge_ohlcv_accumulators(acc_a: dict, acc_b: dict) -> dict:
    if acc_a["event_count"] == 0:
        return acc_b
    if acc_b["event_count"] == 0:
        return acc_a

    merged = _init_ohlcv_accumulator()
    merged["first_ts"] = min(acc_a["first_ts"], acc_b["first_ts"])
    merged["last_ts"] = max(acc_a["last_ts"], acc_b["last_ts"])
    merged["open"] = acc_a["open"] if acc_a["first_ts"] <= acc_b["first_ts"] else acc_b["open"]
    merged["close"] = acc_a["close"] if acc_a["last_ts"] >= acc_b["last_ts"] else acc_b["close"]
    merged["high"] = max(acc_a["high"], acc_b["high"])
    merged["low"] = min(acc_a["low"], acc_b["low"])
    merged["volume_sum"] = acc_a["volume_sum"] + acc_b["volume_sum"]
    merged["event_count"] = acc_a["event_count"] + acc_b["event_count"]
    merged["min_price"] = min(acc_a["min_price"], acc_b["min_price"])
    merged["max_price"] = max(acc_a["max_price"], acc_b["max_price"])
    merged["last_event_ts_ms"] = max(acc_a["last_event_ts_ms"], acc_b["last_event_ts_ms"])
    return merged


class OHLCVAggregateFunction(AggregateFunction):
    def create_accumulator(self) -> dict:
        return _init_ohlcv_accumulator()

    def add(self, value: dict, accumulator: dict) -> dict:
        return _accumulate_ohlcv_event(accumulator, value)

    def get_result(self, accumulator: dict) -> dict:
        return accumulator

    def merge(self, acc_a: dict, acc_b: dict) -> dict:
        return _merge_ohlcv_accumulators(acc_a, acc_b)


class OHLCVWindowProcessFunction(ProcessWindowFunction):
    def __init__(self, is_partial: bool) -> None:
        self.is_partial = is_partial

    def process(self, key, context, elements):
        for acc in elements:
            if acc["event_count"] == 0:
                continue
            yield {
                "ticker": key,
                "window_start_ms": context.window().start,
                "window_end_ms": context.window().end,
                "open": float(acc["open"]),
                "high": float(acc["high"]),
                "low": float(acc["low"]),
                "close": float(acc["close"]),
                "volume": float(acc["volume_sum"]),
                "min_price": float(acc["min_price"]),
                "max_price": float(acc["max_price"]),
                "event_count": int(acc["event_count"]),
                "last_event_ts_ms": int(acc["last_event_ts_ms"]),
                "is_partial": self.is_partial,
            }


class IndicatorEnrichmentProcess(KeyedProcessFunction):
    """Maintain close history per ticker and enrich OHLCV windows with technical indicators."""

    def open(self, runtime_context: RuntimeContext) -> None:
        ttl_days = _env_int("FLINK_INDICATOR_STATE_TTL_DAYS", 90)
        self.max_history_days = _env_int("FLINK_INDICATOR_HISTORY_MAX_DAYS", 120)
        ttl_config = (
            StateTtlConfig.new_builder(FlinkTime.hours(ttl_days * 24))
            .update_ttl_on_create_and_write()
            .never_return_expired()
            .build()
        )
        state_desc = ValueStateDescriptor(
            "indicator_close_history_state",
            Types.PICKLED_BYTE_ARRAY(),
        )
        state_desc.enable_time_to_live(ttl_config)
        self.indicator_close_history_state = runtime_context.get_state(state_desc)

    def process_element(self, value: dict, ctx: "KeyedProcessFunction.Context"):
        history = build_indicator_history(self._load_history(), value, self.max_history_days)
        self.indicator_close_history_state.update(pickle.dumps(history))
        yield enrich_record_with_indicators(value, history)

    def _load_history(self) -> list[dict]:
        encoded = self.indicator_close_history_state.value()
        if encoded is None:
            return []
        try:
            return pickle.loads(encoded)
        except Exception:
            logger.warning("Failed to decode indicator close history state, resetting state for key.")
            return []


class ManageTickerStateProcess(KeyedProcessFunction):
    """
    Task 1.4: manage per-ticker states with TTL.
    - ValueState[List[OHLCV]] stored as pickled bytes.
    - MapState[str, float] for cached indicator values.
    """

    def open(self, runtime_context: RuntimeContext) -> None:
        ttl_hours = _env_int("FLINK_STATE_TTL_HOURS", 24)
        self.max_events = _env_int("FLINK_OHLCV_STATE_MAX_EVENTS", 2000)
        ttl_config = (
            StateTtlConfig.new_builder(FlinkTime.hours(ttl_hours))
            .update_ttl_on_create_and_write()
            .never_return_expired()
            .build()
        )

        ohlcv_state_desc = ValueStateDescriptor(
            "ohlcv_buffer_state",
            Types.PICKLED_BYTE_ARRAY(),
        )
        ohlcv_state_desc.enable_time_to_live(ttl_config)
        self.ohlcv_buffer_state = runtime_context.get_state(ohlcv_state_desc)

        indicator_state_desc = MapStateDescriptor(
            "cached_indicator_state",
            Types.STRING(),
            Types.DOUBLE(),
        )
        indicator_state_desc.enable_time_to_live(ttl_config)
        self.cached_indicator_state = runtime_context.get_map_state(indicator_state_desc)

    def process_element(self, value: dict, ctx: "KeyedProcessFunction.Context"):
        buffer = self._load_buffer()
        buffer.append(
            {
                "date": value["date"],
                "event_ts_ms": int(value["event_ts_ms"]),
                "open": float(value["open"]),
                "high": float(value["high"]),
                "low": float(value["low"]),
                "close": float(value["close"]),
                "volume": float(value["volume"]),
            }
        )
        if len(buffer) > self.max_events:
            buffer = buffer[-self.max_events :]
        self.ohlcv_buffer_state.update(pickle.dumps(buffer))

        # Placeholder cache values for Task 1.5/Task 4 indicator computations.
        self.cached_indicator_state.put("last_close", float(value["close"]))
        self.cached_indicator_state.put("last_volume", float(value["volume"]))
        self.cached_indicator_state.put("last_event_ts_ms", float(value["event_ts_ms"]))

        yield value

    def _load_buffer(self) -> list:
        encoded = self.ohlcv_buffer_state.value()
        if encoded is None:
            return []
        try:
            return pickle.loads(encoded)
        except Exception:
            logger.warning("Failed to decode ohlcv buffer state, resetting state for key.")
            return []


class FeatureSequenceBufferProcess(KeyedProcessFunction):
    """Maintain the latest LSTM feature vectors per ticker for downstream inference."""

    def open(self, runtime_context: RuntimeContext) -> None:
        metric_group = runtime_context.get_metric_group()
        self.sequence_records_total = metric_group.counter("sequence_records_total")
        self.sequence_ready_total = metric_group.counter("sequence_ready_total")
        self.sequence_not_ready_total = metric_group.counter("sequence_not_ready_total")
        self.sequence_imputed_rows_total = metric_group.counter("sequence_imputed_rows_total")

        ttl_days = _env_int("FLINK_FEATURE_BUFFER_TTL_DAYS", 90)
        self.sequence_length = _env_int("LSTM_SEQUENCE_LENGTH", 30)
        ttl_config = (
            StateTtlConfig.new_builder(FlinkTime.hours(ttl_days * 24))
            .update_ttl_on_create_and_write()
            .never_return_expired()
            .build()
        )
        state_desc = ValueStateDescriptor(
            "feature_sequence_buffer_state",
            Types.PICKLED_BYTE_ARRAY(),
        )
        state_desc.enable_time_to_live(ttl_config)
        self.feature_sequence_buffer_state = runtime_context.get_state(state_desc)

    def process_element(self, value: dict, ctx: "KeyedProcessFunction.Context"):
        buffer = self._load_buffer()
        buffer = append_feature_vector_to_buffer(buffer, value, self.sequence_length)
        self.feature_sequence_buffer_state.update(pickle.dumps(buffer))
        enriched = with_sequence_metadata(value, buffer, self.sequence_length)
        self._record_sequence_metrics(enriched)
        yield enriched

    def _record_sequence_metrics(self, record: dict) -> None:
        metrics = sequence_metric_classification(record)
        self.sequence_records_total.inc(metrics["records"])
        self.sequence_ready_total.inc(metrics["ready"])
        self.sequence_not_ready_total.inc(metrics["not_ready"])
        if metrics["imputed_rows"] > 0:
            self.sequence_imputed_rows_total.inc(metrics["imputed_rows"])

    def _load_buffer(self) -> list[dict]:
        encoded = self.feature_sequence_buffer_state.value()
        if encoded is None:
            return []
        try:
            return pickle.loads(encoded)
        except Exception:
            logger.warning("Failed to decode feature sequence buffer state, resetting state for key.")
            return []


def build_event_time_watermark_strategy() -> WatermarkStrategy:
    max_out_of_order_sec = _env_int("FLINK_WATERMARK_OUT_OF_ORDER_SEC", 120)
    return (
        WatermarkStrategy.for_bounded_out_of_orderness(
            Duration.of_seconds(max_out_of_order_sec)
        )
        .with_timestamp_assigner(lambda event, _: int(event["event_ts_ms"]))
    )


def apply_windowing_strategy(parsed_stream):
    window_mode = _env_str("FLINK_WINDOW_MODE", "sliding").lower()
    window_size_ms = _env_int("FLINK_WINDOW_SIZE_MS", 86400000)  # 1 day
    window_slide_ms = _env_int("FLINK_WINDOW_SLIDE_MS", 300000)  # 5 min

    keyed = parsed_stream.key_by(lambda event: event["ticker"], key_type=Types.STRING())

    if window_mode == "tumbling":
        return keyed.window(TumblingEventTimeWindows.of(Time.milliseconds(window_size_ms)))

    return keyed.window(
        SlidingEventTimeWindows.of(
            Time.milliseconds(window_size_ms),
            Time.milliseconds(window_slide_ms),
        )
    )


def is_sliding_mode() -> bool:
    return _env_str("FLINK_WINDOW_MODE", "sliding").lower() != "tumbling"


def main() -> None:
    env = build_env()
    source = build_kafka_source()
    raw_stream = env.from_source(
        source=source,
        watermark_strategy=WatermarkStrategy.no_watermarks(),
        source_name="kafka-stock-raw-ohlcv",
    )
    parsed_stream = raw_stream.map(ParseOHLCVEventMap())
    valid_events = parsed_stream.filter(lambda event: event is not None)
    state_managed_events = (
        valid_events.key_by(lambda event: event["ticker"], key_type=Types.STRING())
        .process(ManageTickerStateProcess())
        .name("manage-ticker-state")
    )
    event_time_events = state_managed_events.assign_timestamps_and_watermarks(
        build_event_time_watermark_strategy()
    )

    windowed = apply_windowing_strategy(event_time_events)
    aggregated = windowed.aggregate(
        OHLCVAggregateFunction(),
        OHLCVWindowProcessFunction(is_partial=is_sliding_mode()),
    )
    aggregated.name("ohlcv-window-aggregation")
    enriched_features = (
        aggregated.key_by(lambda event: event["ticker"], key_type=Types.STRING())
        .process(IndicatorEnrichmentProcess())
        .name("indicator-feature-enrichment")
    )
    sequenced = (
        enriched_features.key_by(lambda event: event["ticker"], key_type=Types.STRING())
        .process(FeatureSequenceBufferProcess())
        .name("feature-sequence-buffer")
    )
    predictions = sequenced.map(LSTMInferenceMapFunction()).name("lstm-inference")
    predictions.print()

    env.execute("stock-analysis-pipeline")


if __name__ == "__main__":
    main()
