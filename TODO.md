# Stock Analysis Pipeline - TODO & Task Breakdown

Tài liệu này theo dõi trạng thái thực tế của repo hiện tại. Quy ước:

- `[x]`: đã có code/test hoặc đã được ghi nhận rõ trong `JOB.md`.
- `[ ]`: còn là placeholder, `EmptyOperator`, file rỗng, stub `NotImplemented`, hoặc mới là mục tiêu vận hành/dashboard chưa có artifact.
- Tên index canonical trong code hiện tại: `stock-engineered-features` và `stock-predictions`.

---

## 1. Thiết kế PyFlink Job (Windowing, State)

**Mục tiêu**: Tạo Flink job xử lý streaming data từ Kafka cho real-time feature pipeline và real-time prediction serving path.

### Công việc cụ thể:

- [x] **1.1 Setup môi trường Flink**
  - Tạo `src/flink/main.py` với Flink `StreamExecutionEnvironment`.
  - Cấu hình job parameters qua env vars: parallelism, checkpoint interval.
  - Thêm Kafka source connector configuration.
  - Ghi nhận trong `JOB.md` là Done.

- [x] **1.2 Định nghĩa Kafka Consumer Source**
  - Tạo `KafkaSource` đọc từ topic `stock.raw.ohlcv`.
  - Parse raw JSON thành Python dict chuẩn hóa trong `src/flink/parsing.py`.
  - Xử lý timestamp từ event (`date` field) thành `event_ts_ms`.
  - Error handling cho corrupt records bằng skip + warning/counters để không làm vỡ stream.

- [x] **1.3 Cấu hình Windowing Strategy**
  - Sliding event-time window mặc định: 1 day window, 5 min slide.
  - Có tumbling fallback qua env cho daily reporting.
  - Watermark strategy dùng bounded out-of-orderness.
  - Window trigger dùng event-time default.

- [x] **1.4 Implement State Management**
  - Có per-ticker state process trong `src/flink/main.py`.
  - Dùng state TTL cho OHLCV/history state.
  - Có cached values scaffold cho indicator computations.
  - RocksDB/state backend production được để dưới cấu hình runtime.

- [x] **1.5 Aggregation Logic**
  - Aggregate theo `ticker`.
  - Tính OHLCV trong window: earliest open, max high, min low, latest close, summed volume.
  - Emit metadata: `window_start_ms`, `window_end_ms`, `event_count`, `min_price`, `max_price`, `last_event_ts_ms`, `is_partial`.
  - Có test logic tại `tests/test_flink_aggregation.py`.

---

## 2. Airflow DAG - Periodic LSTM Training Pipeline

**Mục tiêu**: Tạo Airflow DAG để train LSTM model định kỳ hàng ngày (daily retraining) sử dụng dữ liệu từ Elasticsearch.

**Architecture**:

```text
Elasticsearch (stock-engineered-features)
  -> Airflow DAG: train_lstm_pipeline
     -> Extract data from ES (2.2)
     -> Create time-series dataset (2.3)
     -> Train LSTM model (2.4, 2.5)
     -> Evaluate metrics (2.6)
     -> Convert to ONNX (2.7)
     -> Register to MLflow (2.8)
     -> Promote model to production (2.9)
     -> Notify Flink to reload (2.10)
```

### Công việc cụ thể:

- [x] **2.1 Setup Airflow Environment**
  - Có Airflow services trong `docker-compose.yml`.
  - Metadata backend: PostgreSQL riêng cho Airflow.
  - Có folders `dags/`, `logs/`, `plugins/`.
  - Có DAG skeleton `train_lstm_pipeline` schedule `@daily`.
  - Có bootstrap connections trong `scripts/init_airflow_connections.sh`.

- [x] **2.2 Elasticsearch Data Extractor**
  - Có `dags/tasks/es_data_extractor.py` với `ESDataExtractor`.
  - Query `stock-engineered-features` theo ticker + lookback days.
  - Có pagination, validation, và parquet output.
  - DAG task `extract_features` đã dùng `PythonOperator`.

- [x] **2.3 Time-Series Dataset Creator**
  - Có `dags/tasks/dataset_creator.py` với `LSTMDatasetCreator`.
  - Tạo sliding windows 30 ngày, feature columns + `is_imputed`.
  - Target hiện tại là next-day log-return, không phải raw next-day close.
  - Có `StandardScaler`, train/val/test split, quality filters.
  - Output artifact `.pkl`: `data/lstm_dataset_{ticker}_{date}.pkl`.
  - DAG task `create_dataset` đã dùng `PythonOperator`.

- [ ] **2.4 LSTM Model Architecture**
  - Pending: chưa có LSTM implementation thật trong DAG.
  - Framework target: TensorFlow 2.14+ hoặc PyTorch 2.0+.
  - Model target:
    ```text
    Input: (batch, 30 days, n_features)
    -> LSTM(64, return_sequences=True) + Dropout(0.2)
    -> LSTM(32, return_sequences=False) + Dropout(0.2)
    -> Dense(16, activation='relu')
    -> Dense(1)
    ```
  - Lưu ý: `src/models/trainer.py` hiện còn Random Forest/stub `NotImplemented`, không tính là task LSTM hoàn tất.

- [ ] **2.5 Training Task Implementation**
  - Pending: `train_lstm` trong `dags/train_lstm_pipeline.py` còn là `EmptyOperator`.
  - Cần thay bằng task train thật đọc dataset artifact từ `2.3`.
  - Cần save model artifact và log metrics/artifacts cho downstream.

- [ ] **2.6 Model Evaluation & Metrics**
  - Pending: `evaluate` trong DAG còn là `EmptyOperator`.
  - Cần calculate MSE, MAE, RMSE, MAPE trên test set.
  - Cần baseline comparison và validation criteria.
  - Cần log metrics vào Airflow/MLflow.

- [ ] **2.7 ONNX Conversion**
  - Pending: `convert_onnx` trong DAG còn là `EmptyOperator`.
  - Cần convert LSTM model sang ONNX.
  - Cần verify sample inference và compare framework output vs ONNX output.
  - Output target: `models/lstm_{version}.onnx`.

- [ ] **2.8 MLflow Registration**
  - Pending: `register_mlflow` trong DAG còn là `EmptyOperator`.
  - Cần log parameters, metrics, ONNX file, scaler, và training history.
  - Cần register model name `lstm_stock_predictor`, stage/alias phù hợp.

- [ ] **2.9 Model Promotion**
  - Pending: `promote_model` trong DAG còn là `EmptyOperator`.
  - Cần promote validated model sang production.
  - Cần push `model_version` vào XCom để `2.10` dùng.
  - Cần giữ previous production model làm rollback candidate.

- [x] **2.10 Notification & Model Reload**
  - `notify_flink_reload` đã là `PythonOperator`.
  - Gửi HTTP POST JSON tới endpoint reload qua Airflow connection `flink_default`.
  - Payload có `version`, `model_name`, `stage`, `dag_id`, `run_id`, `task_id`.
  - Có fallback version từ `run_id` nếu `2.9` chưa có XCom `model_version`.
  - Non-2xx response hoặc connection error làm task fail để Airflow retry/surface lỗi.

---

## 3. Tích hợp LSTM ONNX vào Flink (Real-time Inference)

**Mục tiêu**: Load LSTM ONNX model từ MLflow và chạy inference trong Flink streaming job.

### Công việc cụ thể:

- [x] **3.1 LSTM ONNX Inference Wrapper**
  - Có wrapper độc lập tại `src/flink/lstm_onnx_predictor.py`:
    ```python
    class LSTMONNXPredictor:
        def __init__(self, model_path: str, scaler_path: str)
        def predict_sequence(self, features_seq: np.ndarray) -> float
        def validate_sequence(self, seq: np.ndarray) -> bool
    ```
  - Lazy-load ONNX Runtime session và pickle-compatible scaler artifact.
  - Hỗ trợ inject fake session/scaler để unit test không cần model thật.
  - Validate shape `(30, n_features)` hoặc `(1, 30, n_features)`, numeric dtype, no NaN/inf.
  - Output là raw model scalar `float`; conversion sang predicted close thuộc task `3.4`.

- [x] **3.2 Sequence Buffer per Ticker**
  - Có `FeatureSequenceBufferProcess` trong `src/flink/main.py`.
  - Helper testable nằm ở `src/flink/sequence_buffer.py`.
  - `ValueState` duy trì latest sequence per ticker.
  - Env vars: `LSTM_SEQUENCE_LENGTH`, `FLINK_FEATURE_BUFFER_TTL_DAYS`.
  - Handle missing days bằng forward-fill và `is_imputed`.
  - Output metadata: `sequence_ready`, `sequence_length`, `feature_sequence`, `feature_columns`.

- [x] **3.3 Model Loader RichMapFunction**
  - Có scaffold loader tại `src/flink/model_loader.py`.
  - Load `LSTMONNXPredictor` từ local ONNX/scaler artifact paths.
  - Hỗ trợ optional MLflow artifact download qua `MLFLOW_MODEL_URI`.
  - Có safe replacement helper để preserve previous loaded model khi candidate load fail.
  - Hot-reload manager đã hoàn tất ở task `3.6`.

- [x] **3.4 Inference Integration**
  - Có `run_lstm_inference()` để gọi predictor qua `safe_inference_record`.
  - Có `LSTMInferenceMapFunction` load model và kiểm tra hot reload trước prediction.
  - Incomplete sequence hoặc model failure trả structured prediction record thay vì làm vỡ stream.
  - Output tạm đặt `type="prediction"` và `inference_time_ms` để metrics/sink routing dùng.
  - Full prediction schema polish đã hoàn tất ở task `3.5`.

- [x] **3.5 Prediction Output**
  - Có formatter `format_prediction_output()` cho success/error/null predictions.
  - Canonical fields: `type`, `ticker`, `date`, `prediction_date`, `@timestamp`, `actual_close`, `predicted_close`, `model_version`, `confidence`, `inference_time_ms`, `prediction_error`, `error_type`, `sequence_ready`.
  - `prediction_date` dùng value có sẵn hoặc next UTC calendar day từ `date`/`@timestamp`.
  - Output tương thích ES routing vì có `type="prediction"` và `prediction_date`.
  - Market-calendar-aware prediction date và denormalization vẫn để task/model contract sau.

- [x] **3.6 Model Versioning & Hot-Reload**
  - Có `ModelReloadManager` polling theo `FLINK_MODEL_RELOAD_INTERVAL_SECONDS`.
  - Env `FLINK_MODEL_RELOAD_ENABLED` bật/tắt hot reload, default enabled.
  - Artifact signature gồm model version, path, local file mtime và optional MLflow URI.
  - Candidate model phải load thành công trước khi swap.
  - Reload failure preserve previous loaded model và log lỗi.
  - Không implement Flink HTTP receiver; Airflow notification endpoint vẫn là external deployment contract.

- [x] **3.7 Monitoring Metrics**
  - Có `src/flink/monitoring.py` với helper latency bucket, model age, prediction error, sequence metrics.
  - `FeatureSequenceBufferProcess` có counters cho sequence readiness/imputed rows.
  - Có reusable `InferenceMetricsRecorder` cho task `3.4`.
  - Prometheus/CloudWatch exporter không hard-code trong app; để Flink runtime reporter config hoặc future deployment task.

- [x] **3.8 Error Handling**
  - Có `src/flink/error_handling.py` với reusable helpers.
  - Incomplete sequence trả low-confidence/null prediction.
  - Prediction failure trả structured null prediction với `prediction_error`.
  - Có retry with exponential backoff và preserve previous model helper.
  - Chưa wire vào live inference graph; task `3.4` sẽ dùng.

---

## 4. Xử lý logic tính toán Indicator (RSI, MA, EMA, MACD, Volatility)

**Mục tiêu**: Tính toán technical indicators trong Flink.

### Công việc cụ thể:

- [x] **4.1 Tạo Indicator Calculator Module**
  - Có `src/flink/indicators.py` với `IndicatorCalculator`.
  - Có helpers: SMA, SMA bundle, EMA, RSI, MACD, volatility, daily return, lag features.
  - Chính sách thiếu dữ liệu: trả `NaN` để không làm vỡ realtime stream.
  - Invalid period raise `ValueError`.

- [x] **4.2 Implement SMA (Simple Moving Average)**
  - Có `calculate_sma(prices, period)`.
  - Có `calculate_sma_bundle(prices)` trả `sma_10`, `sma_20`, `sma_50`.
  - Thiếu dữ liệu trả `NaN`.
  - Có tests cho sufficient data, insufficient data, latest-window behavior, invalid period.

- [x] **4.3 Implement EMA (Exponential Moving Average)**
  - Có `calculate_ema(prices, period)`.
  - Dùng recursive formula với `alpha = 2 / (period + 1)`.
  - Seed bằng SMA của first period.
  - Đây là pure helper; stateful integration thuộc task `4.8`.

- [x] **4.4 Implement RSI (Relative Strength Index)**
  - Period mặc định 14.
  - Dùng Wilder smoothing.
  - Range hợp lệ 0-100.
  - Có edge-case behavior cho gain-only/loss-only/flat series.

- [x] **4.5 Implement MACD (Moving Average Convergence Divergence)**
  - Có MACD Line = EMA(12) - EMA(26).
  - Có Signal Line = EMA(9) của MACD series.
  - Có Histogram = MACD Line - Signal Line.
  - Đây là pure helper; stateful integration thuộc task `4.8`.

- [x] **4.6 Implement Volatility (Rolling Standard Deviation)**
  - Có `calculate_volatility(prices, period=20)`.
  - Tính latest 20 daily returns.
  - Dùng sample standard deviation (`ddof=1`) để khớp pandas rolling std.
  - Previous close bằng 0 hoặc thiếu dữ liệu trả `NaN`.
  - Stateful close-history integration thuộc task `4.8`.

- [x] **4.7 Lag Features & Daily Return**
  - Có `calculate_daily_return(prices)`.
  - Có `calculate_lag_features(prices)` trả `close_lag_1`, `close_lag_5`, `daily_return`.
  - Thiếu dữ liệu hoặc previous close bằng 0 trả `NaN`.
  - Stateful close-history integration thuộc task `4.8`.

- [x] **4.8 Integration vào Flink Pipeline**
  - Có `IndicatorEnrichmentProcess` trong Flink pipeline sau OHLCV aggregation và trước sequence buffer.
  - Maintain per-ticker close history/state đủ cho SMA/EMA/RSI/MACD/volatility/lag.
  - Dedupe cùng ticker/window timestamp và giữ event mới nhất.
  - Thêm `date`, `@timestamp`, và `is_imputed` để downstream feature/ES path dùng ổn định.
  - Emit enriched feature record:
    ```json
    {
      "ticker": "AAPL",
      "date": "2025-01-15",
      "open": 150.0,
      "high": 155.0,
      "low": 148.0,
      "close": 154.0,
      "volume": 1000000,
      "sma_10": 151.5,
      "sma_20": 150.2,
      "sma_50": 149.8,
      "ema_20": 151.8,
      "rsi": 65.3,
      "macd": 1.5,
      "macd_signal": 1.2,
      "macd_hist": 0.3,
      "volatility": 0.025,
      "close_lag_1": 152.5,
      "close_lag_5": 145.0,
      "daily_return": 0.0131
    }
    ```
  - Có tests cho schema, insufficient history `NaN`, duplicate window, zero previous close, và sequence-buffer compatibility.

---

## 5. Sink dữ liệu kết quả vào Elasticsearch

**Mục tiêu**: Lưu trữ kết quả features và predictions vào Elasticsearch indices.

### Công việc cụ thể:

- [x] **5.1 Cấu hình Elasticsearch Sink**
  - Canonical module: `src/flink/es_sink.py`.
  - Dùng Python Elasticsearch client (`elasticsearch==8.11.0`), không tạo duplicate sink module khác.
  - Config từ env: host, port, scheme, auth, batch, timeout, compression, pool, index names.
  - Batch defaults: size `100`, flush interval `5` seconds.
  - Có validation cho scheme/port/batch/flush/pool/timeout.

- [x] **5.2 Create ElasticsearchSinkFunction**
  - Có Python-side wrapper `ElasticsearchSinkFunction`.
  - Format record thành Elasticsearch document với normalized ISO `@timestamp`.
  - Document ID strategy: `{ticker}_{date}` hoặc `{ticker}_{prediction_date}` khi đủ field.
  - Bulk indexing có retry tối đa 3 lần với exponential backoff.
  - Chưa phải Java/Flink connector implementation.

- [x] **5.3 Implement Dual-Index Sink Strategy**
  - Có helper `route_document_type(record)`.
  - Explicit `type="prediction"` hoặc marker fields route vào `stock-predictions`.
  - Explicit `type="features"` hoặc ambiguous feature records default vào `stock-engineered-features`.
  - Có `DualIndexElasticsearchSink` dùng 2 `ElasticsearchSinkFunction` nội bộ.
  - Feature/prediction buffers flush độc lập.

- [x] **5.4 Mapping & Index Management**
  - Verify/create indices: `stock-engineered-features`, `stock-predictions`.
  - Confirm mappings chứa required fields.
  - Settings: 3 shards, 1 replica, `refresh_interval = 30s`.
  - Reuse shared helpers trong `src/elasticsearch/init_indices.py`.

- [x] **5.5 Error Handling & Monitoring**
  - Log failed inserts với index, document id, source, error reason, timestamp, operation.
  - Metrics: `documents_inserted`, `insert_errors`, `bulk_flushes`, `bulk_failures`, latency tracking.
  - File-based DLQ default: `logs/elasticsearch_deadletter.jsonl`.
  - Circuit breaker mở sau consecutive flush failures và fail fast khi open.
  - Kafka DLQ/separate topic chưa implemented.

- [x] **5.6 Performance Optimization**
  - Bulk indexing qua Python Elasticsearch bulk helper.
  - Batch size, request timeout, compression, pool settings lấy từ config/env.
  - Connection pooling dùng `connections_per_node=config.max_pool_size`.
  - Cluster health check dùng timeout cấu hình được.
  - Real throughput benchmark nằm ở `5.8`.

- [x] **5.7 Data Lifecycle Management**
  - Có ILM/template helpers cho feature và prediction indices.
  - Có daily date-suffixed physical indices, ví dụ `stock-engineered-features-2025-01-15`.
  - Retention default `90d`.
  - Cold-tier/archive behavior phụ thuộc Elasticsearch cluster deployment; code hiện chỉ tạo ILM-compatible policy/template.

- [x] **5.8 Validation & Testing**
  - Có unit tests mocked cho sink helpers.
  - Có opt-in integration test: `tests/test_es_sink_integration.py`.
  - Có smoke validation CLI: `scripts/validate_es_sink.py`.
  - Có lightweight throughput harness: `scripts/benchmark_es_sink.py`.
  - `1000 events/sec` là acceptance target cho benchmark, không phải kết quả đã được chứng minh trong mọi môi trường.

---

## 6. Raw Data Consumer / Raw Elasticsearch Storage

**Mục tiêu**: Theo dõi nhánh A trong blueprint: đọc raw OHLCV và lưu nguyên bản vào Elasticsearch để audit, tracing, và offline training.

### Công việc cụ thể:

- [ ] **6.1 Consume raw OHLCV từ Kafka branch A**
  - Tạo hoặc xác nhận consumer riêng đọc topic `stock.raw.ohlcv`.
  - Không trộn với Flink feature/inference branch nếu mục tiêu là raw audit path.

- [ ] **6.2 Index raw docs vào `stock-raw-ohlcv`**
  - Dùng mapping hiện có `src/elasticsearch/stock_raw_ohlcv_mapping.json`.
  - Ghi raw OHLCV document giữ nguyên field gốc cần audit.

- [ ] **6.3 Idempotent document ID cho raw data**
  - Dùng `_id` ổn định theo `ticker` + `date` hoặc event timestamp.
  - Chống duplicate khi ingestion chạy lại hoặc streaming lấy 7 ngày gần nhất.

- [ ] **6.4 Smoke query + validation raw index**
  - Verify docs retrievable từ `stock-raw-ohlcv`.
  - Kiểm tra count, mapping, sample query theo ticker/date.

---

## 7. Kibana Visualization / Dashboard Templates

**Mục tiêu**: Theo dõi visualization layer cho dashboard realtime trong Kibana.

### Công việc cụ thể:

- [ ] **7.1 Kibana data views**
  - Tạo data views cho `stock-raw-ohlcv`, `stock-engineered-features-*`, `stock-predictions-*`.

- [ ] **7.2 OHLCV / Candlestick Dashboard**
  - Hiển thị OHLCV/candlestick theo ticker và time range.

- [ ] **7.3 Technical Indicators Dashboard**
  - Hiển thị SMA, EMA, RSI, MACD, volatility, daily return.

- [ ] **7.4 Actual vs Predicted Dashboard**
  - Hiển thị actual close vs predicted close.
  - Filter theo ticker, model version, prediction error.

- [ ] **7.5 Dashboard export/import**
  - Export dashboard template NDJSON hoặc document manual setup.
  - Ghi rõ Kibana validation commands/workflow.

---

## 8. Deployment / Architecture Validation

**Mục tiêu**: Theo dõi phần deployment và kiến trúc vận hành được mô tả trong báo cáo.

### Công việc cụ thể:

- [x] **8.1 Kafka 3-node KRaft docker-compose config**
  - `docker-compose.yml` có 3 broker/controller nodes, no ZooKeeper.
  - Có replication factor và min in-sync replicas config.
  - Có `kafka-init` tạo topic `stock.raw.ohlcv`.

- [x] **8.2 Flink Application Mode runbook/validation**
  - Có runbook `docs/flink_application_mode_runbook.md`.
  - Document build/start bằng `stock-jobmanager` và `stock-taskmanager`.
  - Smoke validation JobManager/TaskManager + submitted job qua Flink REST API.
  - `scripts/check_e2e_stack.py` có flag `--require-flink-job`.

- [x] **8.3 End-to-end Docker startup checklist**
  - Có runbook `docs/end_to_end_startup_checklist.md`.
  - Có script read-only `scripts/check_e2e_stack.py`.
  - Checklist cover Kafka, ingestion, Flink, Elasticsearch, Kibana, Airflow/MLflow.
  - Smoke validation có endpoint checks, Kafka topic describe và optional sample ticker search.

---

## Implementation Priority & Phased Approach

### Architecture Overview

```text
STREAMING LAYER (Realtime serving path, Kappa-style)
Kafka (stock.raw.ohlcv)
  -> Flink Job
     -> Sliding window aggregation
     -> Technical indicators
     -> Sequence buffer
     -> LSTM inference
     -> Elasticsearch feature/prediction sink

RAW AUDIT LAYER
Kafka (stock.raw.ohlcv)
  -> Raw data consumer
     -> Elasticsearch raw index: stock-raw-ohlcv

TRAINING LAYER (Scheduled batch/nearline retraining)
Elasticsearch (stock-engineered-features)
  -> Airflow DAG: train_lstm_pipeline
     -> Extract data
     -> Create dataset
     -> Train/evaluate LSTM
     -> Convert ONNX
     -> Register/promote MLflow model
     -> Notify Flink reload
```

### Phase 1 (Weeks 1-2): Core Flink + Feature/Sink Foundations

- ✅ Task 1.1-1.5: Flink setup + Kafka source + windowing + aggregation
- ✅ Task 4.1-4.7: Pure indicator calculations
- ✅ Task 4.8: Stateful indicator integration into Flink pipeline
- ✅ Task 5.1-5.8: Elasticsearch sink setup, routing, lifecycle, validation tooling

### Phase 2 (Weeks 3-4): Training Pipeline

- ✅ Task 2.1-2.3: Airflow setup + ES extraction + dataset creation
- ⏳ Task 2.4-2.9: LSTM architecture, training, evaluation, ONNX, MLflow, promotion
- ✅ Task 2.10: Notification & model reload contract

### Phase 3 (Weeks 5-6): Flink Inference Integration

- ✅ Task 3.2: Sequence buffer
- ✅ Task 3.7: Monitoring helper metrics
- ✅ Task 3.8: Error-handling helpers
- ✅ Task 3.1: ONNX inference wrapper
- ✅ Task 3.3: Model loader scaffold
- ✅ Task 3.4: Inference integration
- ✅ Task 3.5: Prediction output contract
- ✅ Task 3.6: Hot-reload

### Phase 4 (Weeks 7+): Visualization, Raw Branch, Production Hardening

- ⏳ Task 6.1-6.4: Raw data consumer and raw Elasticsearch storage
- ⏳ Task 7.1-7.5: Kibana data views and dashboards
- ✅ Task 8.2: Flink Application Mode runbook and submitted-job validation
- ✅ Task 8.3: End-to-end Docker startup checklist
- [ ] Future: Prometheus/Grafana dashboards
- [ ] Future: Alerting for model drift, inference errors, data quality
- [ ] Future: A/B testing for new models
- [ ] Future: Documentation and operational runbooks

### Key Success Metrics

- **Flink Throughput**: > 1000 events/sec, latency < 500ms p99
- **LSTM Accuracy**: Test RMSE < $2.00, MAE < $1.50
- **Inference**: < 100ms p99, availability > 99.9%
- **Training**: Complete daily retraining in < 2 hours
- **Training Freshness**: Daily retraining completed in scheduled window
- **Data Quality**: > 99% completeness, < 0.1% missing indicators

---

## Current Open Work

- `2.4-2.9`: LSTM model architecture, training, evaluation, ONNX conversion, MLflow registration, promotion.
- `6.1-6.4`: raw data consumer branch and raw Elasticsearch validation.
- `7.1-7.5`: Kibana dashboards and templates.
- Future production hardening: Prometheus/Grafana, alerting, A/B testing, runbooks.

---

## Realtime Classification Note

- Hệ thống là **hybrid**: realtime inference + periodic daily retraining.
- Realtime/Kappa-style áp dụng cho prediction serving path.
- Airflow training path là scheduled batch/nearline retraining, không phải Lambda batch layer cho serving.
- Không đổi tên index trong code; nếu báo cáo dùng `technical_features` hoặc `stock_predictions`, hãy xem đó là alias mô tả, còn canonical indices là `stock-engineered-features` và `stock-predictions`.
