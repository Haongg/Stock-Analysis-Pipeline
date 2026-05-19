# JOB Plan Log

Tài liệu này tóm tắt các plan đã chốt cho từng task (kèm trạng thái thực thi hiện tại).

## Task 1.1 - Setup môi trường Flink
**Status**: Done

### Plan đã chốt
- Tạo `StreamExecutionEnvironment` trong `src/flink/main.py`.
- Cấu hình `parallelism` + `checkpoint interval` bằng env vars.
- Thêm cấu hình Kafka source cơ bản.
- Đảm bảo dependency PyFlink có trong môi trường chạy.

### Kết quả chính
- `main.py` có env setup + checkpoint + Kafka source scaffold.

---

## Task 1.2 - Kafka Consumer Source + Parse JSON
**Status**: Done

### Plan đã chốt
- Dùng `KafkaSource` với `SimpleStringSchema` để nhận raw JSON.
- Parse JSON -> dict chuẩn hóa (`ticker`, `date`, `event_ts_ms`, OHLCV, optional `ingested_at`).
- Xử lý timestamp từ `date` (ISO8601 -> epoch ms).
- Corrupt policy: skip + warning log + counters (`records_total`, `records_valid`, `records_invalid_*`).
- Lọc bỏ record lỗi để giữ stream ổn định.

### Kết quả chính
- Tách parser reusable tại `src/flink/parsing.py`.
- Gắn parser + counters trong Flink map function.

---

## Task 1.3 - Windowing Strategy (Realtime)
**Status**: Done

### Plan đã chốt
- Watermark: `BoundedOutOfOrderness` (default 120s, cấu hình qua env).
- Timestamp assigner lấy từ `event_ts_ms`.
- Window mode mặc định: sliding `1 day / 5 min`.
- Có fallback `tumbling` qua env.
- Giữ `EventTimeTrigger` mặc định.

### Kết quả chính
- Pipeline đã có event-time watermark + window wiring trong `main.py`.

---

## Task 1.4 - State Management
**Status**: Done

### Plan đã chốt
- `ValueState<List<OHLCV>>` để lưu buffer per ticker.
- `MapState<String, Double>` cho cached values.
- TTL 24h cho cả 2 state.
- Bật RocksDB state backend cho production qua cấu hình env.

### Kết quả chính
- Implement `ManageTickerStateProcess` với TTL + state descriptors.
- Gắn vào flow trước bước assign watermark.

---

## Task 1.5 - Aggregation Logic
**Status**: Done

### Plan đã chốt
- Thay summary tạm bằng `AggregateFunction` incremental + `ProcessWindowFunction`.
- Aggregate theo ticker cho OHLCV:
  - `open`: earliest event timestamp
  - `high`: max high
  - `low`: min low
  - `close`: latest event timestamp
  - `volume`: SUM
- Bổ sung fields: `window_start_ms`, `window_end_ms`, `event_count`, `min_price`, `max_price`, `last_event_ts_ms`, `is_partial`.
- Giữ output `print()` tạm để smoke test.

### Kết quả chính
- Implement aggregator + process function trong `main.py`.
- Có test logic tại `tests/test_flink_aggregation.py`.

---

## Task 2.1 - Setup Airflow Environment
**Status**: Done

### Plan đã chốt
- Dùng official image `apache/airflow:2.8.x-python3.11`.
- Metadata backend: PostgreSQL riêng cho Airflow (`airflow-db`).
- Executor: `LocalExecutor`.
- Thêm services: `airflow-init`, `airflow-webserver`, `airflow-scheduler` (profile `airflow`).
- Mount folders `dags/`, `logs/`, `plugins/`.
- Bootstrap admin user + connections (`elasticsearch_default`, `postgres_default`, `mlflow_default`).
- Tạo DAG skeleton `train_lstm_pipeline` schedule `@daily`.

### Kết quả chính
- `docker-compose.yml` đã có full Airflow stack.
- Có script `scripts/init_airflow_connections.sh`.
- Có DAG skeleton tại `dags/train_lstm_pipeline.py`.

---

## Task 2.2 - Elasticsearch Data Extractor
**Status**: Done

### Plan đã chốt
- Tạo `ESDataExtractor` trong `dags/tasks/es_data_extractor.py`.
- Query index `stock-engineered-features` theo `ticker` + lookback days.
- Pagination bằng `search_after`.
- Validate dữ liệu: required columns, record count, null-rate, timestamp continuity.
- Output parquet: `data/features_{ticker}_{date}.parquet`.
- Nối vào DAG bằng `PythonOperator` cho task `extract_features`.

### Kết quả chính
- Extractor + validation + parquet save đã hoạt động trong code.
- DAG `extract_features` không còn là `EmptyOperator`.

---

## Task 2.10 - Notification & Model Reload
**Status**: Done

### Plan đã chốt
- Thay `notify_flink_reload` từ `EmptyOperator` sang `PythonOperator`.
- Gửi HTTP POST tới Flink reload endpoint sau khi `promote_model` thành công.
- Dùng Airflow connection `flink_default` để cấu hình base URL.
- Payload có `version`, `model_name`, `stage`, `dag_id`, `run_id`, `task_id`.
- Non-2xx response hoặc connection error phải fail task để Airflow retry/surface lỗi.

### Kết quả chính
- DAG đã có callable `notify_flink_reload_task` dùng stdlib `urllib`, không cần thêm HTTP provider.
- Có fallback model version từ `run_id` khi `promote_model` chưa push XCom `model_version`.
- `scripts/init_airflow_connections.sh` bootstrap thêm connection `flink_default`.
- `.env.template` và `docker-compose.yml` có default `AIRFLOW_CONN_FLINK_DEFAULT=http://stock-jobmanager:8081`.
- Lưu ý: Flink-side receiver `/flink/model-reload` vẫn là contract cho task model-loader/reload sau.

---

## Task 3.1 - LSTM ONNX Inference Wrapper
**Status**: Done

### Plan đã chốt
- Tạo wrapper ONNX inference độc lập, chưa wire vào live Flink graph.
- Load ONNX Runtime session và scaler artifact khi chạy thật.
- Cho phép inject fake session/scaler để test không cần model production.
- Validate sequence shape, numeric dtype, và NaN/inf trước inference.
- Trả raw model scalar `float`; conversion sang predicted close để task 3.4 xử lý.

### Kết quả chính
- Tạo `src/flink/lstm_onnx_predictor.py` với `LSTMONNXPredictor`.
- Hỗ trợ `predict_sequence()` cho matrix 2D/3D và `predict_record()` cho records từ sequence buffer.
- Default feature order lấy từ `FEATURE_SEQUENCE_COLUMNS`.
- Thêm dependency `onnxruntime` vào `requirements.txt` cho real runtime inference.
- Có test coverage tại `tests/test_lstm_onnx_predictor.py`.

---

## Task 3.2 - Sequence Buffer per Ticker
**Status**: Done

### Plan đã chốt
- Tạo buffer sequence 30 ngày cho mỗi ticker bằng Flink `ValueState`.
- Dùng state TTL 90 ngày qua env `FLINK_FEATURE_BUFFER_TTL_DAYS`.
- Dùng env `LSTM_SEQUENCE_LENGTH` để cấu hình độ dài sequence, mặc định 30.
- Append feature vector theo ngày, dedupe cùng ngày bằng event timestamp mới nhất.
- Forward-fill ngày bị thiếu và đánh dấu `is_imputed = 1`.
- Expose metadata `sequence_ready`, `sequence_length`, `feature_sequence`, `feature_columns`.

### Kết quả chính
- Implement `FeatureSequenceBufferProcess` trong `src/flink/main.py`.
- Thêm helper testable trong `src/flink/sequence_buffer.py` cho parse date, tạo feature vector, forward-fill, dedupe, trim buffer.
- Buffer giữ đúng thứ tự feature columns theo `LSTMDatasetCreator`.
- Pipeline đã wire step `feature-sequence-buffer` sau OHLCV aggregation để chuẩn bị cho task 3.4.
- Có test coverage tại `tests/test_feature_sequence_buffer.py`.

---

## Task 3.3 - Model Loader RichMapFunction
**Status**: Done

### Plan đã chốt
- Implement model loading scaffold độc lập trong `src/flink/model_loader.py`.
- Resolve local ONNX/scaler artifact paths từ env.
- Hỗ trợ optional MLflow artifact download khi có `MLFLOW_MODEL_URI`.
- Load `LSTMONNXPredictor` và expose predictor cho task 3.4.
- Preserve previous loaded model nếu candidate model load fail.
- Không chạy inference trong stream ở task này; full polling/hot-reload để task 3.6.

### Kết quả chính
- Thêm `ModelArtifactConfig`, `LoadedModel`, `ModelLoadError`.
- Thêm `load_lstm_model`, `resolve_model_artifact_config_from_env`, `resolve_model_artifact_paths`.
- Thêm `ModelLoaderRichMapFunction` với `open`, `map`, `get_loaded_model`, `get_predictor`.
- `.env.template` có env vars cho model/scaler paths, version, MLflow URI và artifact names.
- Có test coverage tại `tests/test_model_loader.py`.

---

## Task 3.4 - Inference Integration
**Status**: Done

### Plan đã chốt
- Wire loaded LSTM predictor vào Flink stream sau `FeatureSequenceBufferProcess`.
- Dùng `safe_inference_record` để incomplete sequence hoặc model failure tạo structured prediction record.
- Record inference metrics bằng `InferenceMetricsRecorder`.
- Không wire Elasticsearch sink, full prediction schema polish hoặc hot reload trong task này.

### Kết quả chính
- Tạo `src/flink/inference_integration.py` với `run_lstm_inference`, `build_model_not_loaded_record`, `LSTMInferenceMapFunction`.
- Pipeline giờ đi theo flow: aggregation -> indicator enrichment -> sequence buffer -> LSTM inference.
- Prediction records có `type="prediction"` và `inference_time_ms`.
- Startup model load failure trả `prediction_error="model_not_loaded"` thay vì crash stream.
- Có test coverage tại `tests/test_flink_inference_integration.py`.

---

## Task 3.5 - Prediction Output
**Status**: Done

### Plan đã chốt
- Finalize prediction record contract emitted by Flink inference path.
- Dùng formatter thuần Python để success/error/null predictions có cùng schema.
- Thêm `prediction_date` để ES routing/daily index logic dùng được.
- Không attach Elasticsearch sink trong task này.

### Kết quả chính
- Thêm `format_prediction_output` trong `src/flink/inference_integration.py`.
- `run_lstm_inference` và `build_model_not_loaded_record` giờ trả canonical prediction records.
- Output có `type="prediction"`, `prediction_date`, `actual_close`, `predicted_close`, `model_version`, `confidence`, `inference_time_ms`, `prediction_error`, `error_type`.
- `prediction_date` derive từ `date`/`@timestamp` bằng next UTC calendar day nếu chưa có sẵn.
- Có test coverage tại `tests/test_flink_inference_integration.py`, bao gồm ES route compatibility.

---

## Task 3.7 - Monitoring Metrics
**Status**: Done

### Plan đã chốt
- Thêm helper monitoring thuần Python cho latency bucket, model age, prediction error, sequence metrics.
- Instrument `FeatureSequenceBufferProcess` bằng Flink counters cho sequence readiness.
- Chuẩn bị reusable recorder cho inference metrics để task 3.4 dùng sau.
- Giữ exporter Prometheus/CloudWatch ở tầng Flink runtime config, không hard-code trong app.

### Kết quả chính
- Tạo `src/flink/monitoring.py` với các helper testable.
- Thêm counters: `sequence_records_total`, `sequence_ready_total`, `sequence_not_ready_total`, `sequence_imputed_rows_total`.
- Thêm `InferenceMetricsRecorder` với counters prediction/error/latency bucket và model-age gauge holder.
- Có test coverage tại `tests/test_monitoring.py`.

---

## Task 3.8 - Error Handling
**Status**: Done

### Plan đã chốt
- Tạo helper error-handling dùng lại cho inference path, chưa wire vào live Flink graph.
- Incomplete sequence trả null prediction với low confidence.
- Prediction failure trả null prediction kèm `prediction_error` và `error_type`.
- Retry callable tối đa 3 lần với exponential backoff.
- Preserve previous model khi candidate model load fail hoặc `None`.

### Kết quả chính
- Tạo `src/flink/error_handling.py` với helper `build_null_prediction`, `mark_low_confidence`, `should_skip_inference`, `retry_with_backoff`, `keep_previous_model`, `safe_inference_record`.
- Error output tương thích với `prediction_has_error` trong monitoring.
- `safe_inference_record` xử lý success/failure/incomplete sequence cho task 3.4 dùng sau.
- Có test coverage tại `tests/test_error_handling.py`.

---

## Task 4.1 - Indicator Calculator Module
**Status**: Done

### Plan đã chốt
- Tạo `IndicatorCalculator` thuần Python trong `src/flink/indicators.py`.
- Cung cấp các hàm core: SMA, EMA, RSI, MACD, Volatility.
- Chính sách thiếu dữ liệu: trả `NaN` để không làm vỡ realtime stream.

### Kết quả chính
- Module indicator đã được tạo và có test cơ bản.

---

## Task 4.2 - Implement SMA (SMA10/SMA20/SMA50)
**Status**: Done

### Plan đã chốt
- Harden contract `calculate_sma(prices, period)`:
  - `period <= 0` -> `ValueError`
  - thiếu dữ liệu -> `NaN`
  - đủ dữ liệu -> mean của cửa sổ mới nhất.
- Thêm helper `calculate_sma_bundle(prices)` trả:
  - `sma_10`, `sma_20`, `sma_50`
- Mở rộng test coverage cho các case đủ dữ liệu/thiếu dữ liệu/window mới nhất/invalid period.

### Kết quả chính
- Đã có SMA bundle cho downstream integration (task 4.8).
- Test cho `SMA10/20/50` và edge cases đã được bổ sung.

---

## Task 4.3 - Implement EMA (Exponential Moving Average)
**Status**: Done

### Plan đã chốt
- Implement EMA(20) trong `IndicatorCalculator`.
- Dùng recursive formula: `EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}`.
- Tính `alpha = 2 / (period + 1)`.
- Bổ sung validation cho period và behavior khi thiếu dữ liệu.

### Kết quả chính
- `calculate_ema(prices, period)` đã được implement trong `src/flink/indicators.py`.
- Hàm có `period` validation (`ValueError` khi `period <= 0`).
- Thiếu dữ liệu trả `NaN`, đúng contract realtime-safe.
- Có test cho EMA behavior trong `tests/test_indicators.py`.

---

## Task 4.4 - Implement RSI (Relative Strength Index)
**Status**: Done

### Plan đã chốt
- Implement RSI period mặc định 14.
- Áp dụng công thức `RSI = 100 - (100 / (1 + RS))`.
- Dùng Wilder smoothing cho average gain/loss.
- Đảm bảo output nằm trong range hợp lệ và xử lý edge cases.

### Kết quả chính
- `calculate_rsi(prices, period=14)` đã được implement trong `src/flink/indicators.py`.
- Có smoothing gain/loss theo Wilder method.
- Có xử lý edge case: zero-loss/gain-only/flat series.
- Có test kiểm tra range và behavior trong `tests/test_indicators.py`.

---

## Task 4.5 - Implement MACD (Moving Average Convergence Divergence)
**Status**: Done

### Plan đã chốt
- Implement MACD Line = `EMA(12) - EMA(26)`.
- Implement Signal Line = `EMA(9)` của MACD series.
- Implement Histogram = `MACD - Signal`.
- Bổ sung guard khi dữ liệu không đủ cho EMA26/signal.

### Kết quả chính
- `calculate_macd(prices)` đã trả tuple `(macd, signal, histogram)` trong `src/flink/indicators.py`.
- Có insufficient-data guard trả `NaN` khi series ngắn.
- Consistency `hist = macd - signal` đã được cover trong tests.
- Có test cho short/long series và trending series trong `tests/test_indicators.py`.

---

## Task 4.6 - Implement Volatility (Rolling Standard Deviation)
**Status**: Done

### Plan đã chốt
- Dùng `calculate_volatility(prices, period=20)` trong `IndicatorCalculator`.
- Tính daily returns bằng `(close_t - close_{t-1}) / close_{t-1}`.
- Dùng latest `period` returns và sample standard deviation (`ddof=1`).
- Thiếu `period + 1` close prices trả `NaN`.
- Previous close bằng 0 trả `NaN`; `period <= 0` raise `ValueError`.

### Kết quả chính
- Volatility implementation đã khớp batch feature pipeline (`pct_change().rolling(...).std()`).
- Bổ sung test exact sample std, latest-window behavior, insufficient data, zero previous close, invalid period.
- Stateful Flink integration vẫn để task 4.8 xử lý.

---

## Task 4.7 - Lag Features & Daily Return
**Status**: Done

### Plan đã chốt
- Thêm helper `calculate_daily_return(prices)` vào `IndicatorCalculator`.
- Thêm helper `calculate_lag_features(prices)` trả `close_lag_1`, `close_lag_5`, `daily_return`.
- `close_lag_1` tương đương pandas `shift(1)`, `close_lag_5` tương đương `shift(5)`.
- `daily_return` tương đương pandas `pct_change()` trên close mới nhất.
- Thiếu dữ liệu hoặc previous close bằng 0 trả `NaN`.

### Kết quả chính
- Pure Flink-side helper đã khớp batch `FeatureEngineer` cho lag features và daily return.
- Bổ sung test exact daily return, missing windows, zero previous close, lag bundle, int/float coercion.
- Stateful Flink integration vẫn để task 4.8 xử lý.

---

## Task 4.8 - Integration vào Flink Pipeline
**Status**: Done

### Plan đã chốt
- Thêm stateful enrichment step sau OHLCV window aggregation và trước `FeatureSequenceBufferProcess`.
- Duy trì close history per ticker bằng Flink `ValueState` với TTL 90 ngày.
- Dùng `IndicatorCalculator` để tính SMA/EMA/RSI/MACD/volatility/lag/daily return.
- Output enriched record phải đủ feature schema cho LSTM sequence buffer.
- Giữ `NaN` policy cho insufficient history, không drop record.

### Kết quả chính
- Implement `IndicatorEnrichmentProcess` trong `src/flink/main.py`.
- Thêm helper module `src/flink/indicator_enrichment.py` với `build_indicator_history` và `enrich_record_with_indicators` để test độc lập.
- Pipeline giờ đi theo flow: aggregation -> indicator enrichment -> sequence buffer.
- Enriched records có `date`, `@timestamp`, indicator fields và `is_imputed`.
- Có test coverage tại `tests/test_flink_indicator_integration.py`.

---

## Task 5.1 - Cấu hình Elasticsearch Sink
**Status**: Done

### Plan đã chốt
- Hoàn thiện config Elasticsearch sink trong `src/flink/es_sink.py`, giữ tên module hiện có.
- Dùng Python Elasticsearch client (`elasticsearch==8.11.0`) đã có trong requirements.
- Đọc host/port/scheme/auth/batch/timeout/index config từ env.
- Batch default: size `100`, flush interval `5` seconds.
- Validate scheme, port, batch size, flush interval và timeout/pool settings.

### Kết quả chính
- `ElasticsearchSinkConfig` có validation cho config không hợp lệ.
- `create_elasticsearch_client` chỉ bật `basic_auth` khi đủ username/password.
- `.env.template` và `src/elasticsearch/README.md` đã bổ sung các biến sink config.
- Test coverage đã mở rộng cho host URL, auth behavior, invalid config, index defaults.

---

## Task 5.2 - Create ElasticsearchSinkFunction
**Status**: Done

### Plan đã chốt
- Tạo generic Python-side sink wrapper trong `src/flink/es_sink.py`.
- Format document với ISO `@timestamp`.
- Stable document ID theo `{ticker}_{date}` hoặc `{ticker}_{prediction_date}` khi đủ field.
- Không có đủ ID field thì để Elasticsearch auto-generate ID.
- Bulk indexing có retry tối đa 3 lần với exponential backoff.
- Chưa wire vào Flink graph; dual-index routing để task 5.3.

### Kết quả chính
- Thêm helper `normalize_timestamp`, `build_document_id`, `bulk_index_documents_with_retry`.
- Thêm class `ElasticsearchSinkFunction` với buffer, `invoke`, `flush`, `close`.
- Test coverage cho timestamp normalization, document ID, retry behavior, buffer flush, close flush.

---

## Task 5.3 - Implement Dual-Index Sink Strategy
**Status**: Done

### Plan đã chốt
- Hoàn thiện dual-index routing trong `src/flink/es_sink.py`.
- Feature records đi vào `ES_FEATURE_INDEX` / `stock-engineered-features`.
- Prediction records đi vào `ES_PREDICTION_INDEX` / `stock-predictions`.
- Reuse `ElasticsearchSinkFunction`, document builders, timestamp normalization và retry bulk indexing từ task 5.2.
- Chưa wire vào Flink graph; pipeline placement và index lifecycle để các task sau xử lý.

### Kết quả chính
- Thêm helper `route_document_type`, `is_feature_record`, `is_prediction_record`.
- Thêm `DualIndexElasticsearchSink` với hai sink nội bộ cho feature và prediction documents.
- Feature/prediction buffers flush độc lập theo batch size và `close()` flush cả hai sink.
- Bổ sung test cho routing rules, configured index names, independent flush và close flush behavior.

---

## Task 5.4 - Mapping & Index Management
**Status**: Done

### Plan đã chốt
- Thêm reusable Elasticsearch index-management helpers trong `src/flink/es_sink.py`.
- Verify/create hai sink indices: `stock-engineered-features` và `stock-predictions`.
- Validate existing mappings có đủ required fields trước khi coi index là usable.
- Align mapping settings theo TODO: `3` shards, `1` replica, `refresh_interval=30s`.
- Cập nhật `src/elasticsearch/init_indices.py` để reuse Flink sink config/client/index helpers.

### Kết quả chính
- Thêm `load_index_mapping`, `create_index_if_missing`, `validate_index_mapping`, `ensure_elasticsearch_indices`.
- Feature/prediction mapping JSON đã thêm missing fields và settings `3/1/30s`.
- Init script vẫn tạo raw OHLCV index, sau đó verify/create feature và prediction indices qua shared helpers.
- Bổ sung mocked tests cho create/skip/validate mapping và expected index settings.

---

## Task 5.5 - Error Handling & Monitoring
**Status**: Done

### Plan đã chốt
- Harden Elasticsearch sink flush path với structured insert errors, optional metrics, file DLQ và lightweight circuit breaker.
- Giữ behavior batching/index routing hiện có của `ElasticsearchSinkFunction` và `DualIndexElasticsearchSink`.
- File DLQ dùng default `logs/elasticsearch_deadletter.jsonl`; Kafka DLQ để task sau nếu cần.

### Kết quả chính
- Thêm `ElasticsearchSinkMetrics` cho counters `documents_inserted`, `insert_errors`, `bulk_flushes`, `bulk_failures` và latency total.
- Thêm `ElasticsearchCircuitBreaker` mở circuit sau exhausted flush failures và cho retry sau reset window.
- Bulk item errors và exhausted exceptions được chuyển thành JSONL dead-letter entries có index, document id, source, error reason, timestamp và operation.
- `flush()` ghi metrics/DLQ, clear buffer sau attempted flush, và re-raise exception để runtime retry/surface lỗi.
- Bổ sung mocked tests cho metrics, partial errors, exception DLQ, circuit breaker và valid JSONL DLQ.

---

## Task 5.6 - Performance Optimization
**Status**: Done

### Plan đã chốt
- Hoàn thiện performance knobs cho Elasticsearch sink ở mức config/helper.
- Giữ bulk indexing qua Python Elasticsearch bulk helper.
- Dùng batch size, request timeout, compression và connection pool settings từ env/config.
- Health check dùng timeout cấu hình được.
- Không làm live throughput benchmark; phần đó thuộc task 5.8.

### Kết quả chính
- `ElasticsearchSinkConfig` đã có defaults cho batch size `100`, flush interval `5s`, pool `4/8`, compression và health timeout.
- `create_elasticsearch_client` truyền `connections_per_node=config.max_pool_size` và `http_compress`.
- Bulk helper truyền `chunk_size=config.batch_size`, `request_timeout=config.request_timeout`, `raise_on_error=False`.
- `.env.template` và Elasticsearch README đã bổ sung pool/health timeout knobs.
- Bổ sung tests cho pool/compression settings, bulk chunk/timeout options, invalid pool/health config và cluster health timeout.

---

## Task 5.7 - Data Lifecycle Management
**Status**: Done

### Plan đã chốt
- Thêm Elasticsearch ILM/template helpers cho feature và prediction indices.
- Dùng daily date-suffixed physical indices như `stock-engineered-features-2025-01-15`.
- Giữ logical base names từ config: `ES_FEATURE_INDEX`, `ES_PREDICTION_INDEX`.
- Retention default `90d`; cold/delete behavior cấu hình qua ILM policy.
- Init script setup lifecycle trước khi verify/create indices.

### Kết quả chính
- Thêm lifecycle config: `ES_ILM_ENABLED`, `ES_ILM_POLICY_NAME`, `ES_INDEX_RETENTION_DAYS`, `ES_DAILY_INDEX_ENABLED`.
- Thêm helper `build_daily_index_name`, `resolve_index_name`, `build_ilm_policy_body`, `build_index_template_body`, `setup_elasticsearch_lifecycle`.
- Bulk actions dùng daily index suffix khi `daily_index_enabled=True`.
- Init script gọi `setup_elasticsearch_lifecycle` trước `ensure_elasticsearch_indices`.
- `.env.template` và Elasticsearch README đã document ILM/daily-index knobs.
- Bổ sung mocked tests cho daily index naming, ILM policy body, index templates và disabled lifecycle path.

---

## Task 5.8 - Validation & Testing
**Status**: Done

### Plan đã chốt
- Thêm validation layer cuối cho Elasticsearch sink gồm unit/static checks, opt-in integration test, smoke validation script và throughput harness.
- Integration/performance checks phải opt-in để local unit test không cần Docker, Elasticsearch hoặc Kibana.
- Kibana verification document theo manual workflow.

### Kết quả chính
- Thêm `tests/test_es_sink_integration.py`, skip mặc định trừ khi `ES_INTEGRATION_TEST=1`.
- Thêm `scripts/validate_es_sink.py` để setup lifecycle/index, ghi feature/prediction sample docs, refresh và query lại theo ticker.
- Thêm `scripts/benchmark_es_sink.py` để index synthetic feature docs và report events/sec, inserted count, error count.
- Elasticsearch README đã document unit, integration, smoke validation, throughput và Kibana data-view checks.
- `src/elasticsearch/init_indices.py` và scripts mới có root path bootstrap để chạy trực tiếp bằng `python3 path/to/script.py`.

---

## Task 3.6 - Model Hot Reload
**Status**: Done

### Plan đã chốt
- Implement polling-based hot reload cho Flink inference path.
- Track artifact signature bằng model version, resolved model/scaler paths, local file mtimes và optional MLflow URI.
- Chỉ reload khi interval đã elapsed và candidate artifact thay đổi.
- Candidate model phải load thành công trước khi swap.
- Nếu reload fail, preserve previous loaded model và tiếp tục stream bằng model cũ.
- Không thêm Flink HTTP receiver; Airflow reload endpoint vẫn là external deployment contract.

### Kết quả chính
- Thêm `ModelReloadManager`, `ModelReloadSettings` và `ModelArtifactSignature` trong `src/flink/model_loader.py`.
- Thêm env `FLINK_MODEL_RELOAD_ENABLED=true` và `FLINK_MODEL_RELOAD_INTERVAL_SECONDS=300`.
- `LSTMInferenceMapFunction` giờ khởi tạo reload manager và gọi `maybe_reload()` trước prediction.
- Startup không load được model vẫn emit canonical `model_not_loaded`; reload fail vẫn giữ model đang chạy.
- Bổ sung tests cho reload interval, signature change, disabled reload, failed reload fallback và inference map dùng reloaded version.

---

## Task 8.3 - End-to-end Docker Startup Checklist
**Status**: Done

### Plan đã chốt
- Thêm runbook checklist cho startup toàn bộ Docker stack theo thứ tự rõ ràng.
- Cover core services, Kafka, Elasticsearch/Kibana, index init, Flink, ingestion và optional Airflow.
- Thêm script smoke validation read-only, không start/stop container.
- Không claim training/model promotion hoàn tất vì `2.4-2.9` vẫn pending.

### Kết quả chính
- Thêm `docs/end_to_end_startup_checklist.md` với prerequisites, startup commands, validation commands và troubleshooting.
- Thêm `scripts/check_e2e_stack.py` để check Elasticsearch, Flink, Kafka topic, optional core/Kibana/Airflow và optional sample ticker query.
- README đã trỏ từ Quick Start sang checklist và ghi rõ caveat `model_not_loaded` khi chưa có ONNX/scaler artifacts.
- `TODO.md` đã mark `8.3` done; Application Mode submitted-job validation đã hoàn tất ở task `8.2`.

---

## Task 8.2 - Flink Application Mode Runbook & Validation
**Status**: Done

### Plan đã chốt
- Thêm runbook riêng cho Flink Application Mode, tách khỏi checklist end-to-end task `8.3`.
- Reuse compose services `stock-jobmanager` và `stock-taskmanager`.
- Giữ entrypoint canonical `standalone-job --python /opt/flink/usrlib/python_code/main.py`.
- Thêm smoke validation cho JobManager, TaskManager và submitted-job visibility.

### Kết quả chính
- Thêm `docs/flink_application_mode_runbook.md` với topology, prerequisites, build/start commands, validation commands và troubleshooting.
- README đã link tới Flink Application Mode runbook ở phần Start Flink.
- `scripts/check_e2e_stack.py` có flag `--require-flink-job` để fail khi không có active submitted job.
- `TODO.md` đã mark `8.2` complete; Application Mode validation không còn nằm trong Current Open Work.

---

## Next suggested tasks
- `2.4` Implement real LSTM model architecture
