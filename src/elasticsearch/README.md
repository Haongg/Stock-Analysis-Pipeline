# Elasticsearch Configuration for Stock Analysis Pipeline

This directory contains Elasticsearch index mappings and configuration for the Stock Analysis Pipeline.

## Index Overview

### 1. `stock-raw-ohlcv`
Stores raw OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.

**Key Fields:**
- `@timestamp`: Trading date (mapped from `date` field)
- `ticker`: Stock symbol (keyword)
- `open`, `high`, `low`, `close`: Price data (float)
- `volume`: Trading volume (long)
- `ingested_at`: Timestamp when data was ingested (date)

### 2. `stock-engineered-features`
Stores processed data with technical indicators and features for ML training.

**Key Fields:**
- `@timestamp`: Trading date
- `ticker`: Stock symbol (keyword)
- Raw OHLCV fields
- Technical indicators: `sma_10`, `sma_20`, `sma_50`, `ema_20`, `rsi`, `macd`, `macd_signal`, `macd_hist`
- Risk metrics: `volatility`, `daily_return`
- Lag features: `close_lag_1`, `close_lag_5`

### 3. `stock-predictions`
Stores ML model predictions and actual results.

**Key Fields:**
- `@timestamp`: Prediction timestamp
- `ticker`: Stock symbol (keyword)
- `predicted_close`: Model prediction (float)
- `actual_close`: Actual closing price (float, optional)
- `model_version`: ML model version (keyword)
- `prediction_date`: Date the prediction is for (date)
- `confidence`: Prediction confidence score (float, optional)

## Index Settings

All indices use:
- **Shards**: 3 (for parallel processing)
- **Replicas**: 1 (for fault tolerance)
- **Codec**: best_compression (for storage efficiency)
- **Refresh Interval**: 30s (balance between real-time and performance)

## Usage

### Creating Indices

Run the initialization script to create all indices:

```bash
cd /path/to/project
python src/elasticsearch/init_indices.py
```

### Environment Variables

Set these environment variables for Elasticsearch connection:

```bash
export ES_HOST=localhost
export ES_PORT=9200
export ES_SCHEME=http
# export ES_USER=elastic
# export ES_PASSWORD=changeme
export ES_BULK_BATCH_SIZE=100
export ES_BULK_FLUSH_INTERVAL_SECONDS=5
export ES_REQUEST_TIMEOUT_SECONDS=30
export ES_CORE_POOL_SIZE=4
export ES_MAX_POOL_SIZE=8
export ES_HTTP_COMPRESS=true
export ES_CLUSTER_HEALTH_TIMEOUT_SECONDS=5
export ES_FEATURE_INDEX=stock-engineered-features
export ES_PREDICTION_INDEX=stock-predictions
export ES_ILM_ENABLED=true
export ES_ILM_POLICY_NAME=stock-analysis-90d-policy
export ES_INDEX_RETENTION_DAYS=90
export ES_DAILY_INDEX_ENABLED=true
```

When daily indexing is enabled, feature and prediction documents are written to
date-suffixed indices such as `stock-engineered-features-2025-01-15` and
`stock-predictions-2025-01-15`. The initialization script installs the ILM
policy and matching index templates for those patterns.

### Integration with Pipeline

To integrate with your data pipeline:

1. **Raw Data Indexing**: After fetching from Yahoo Finance, index to `stock-raw-ohlcv`
2. **Feature Indexing**: After feature engineering, index to `stock-engineered-features`
3. **Prediction Indexing**: After model inference, index predictions to `stock-predictions`

### Kibana Dashboards

Use Kibana to create visualizations:
- Time-series charts for stock prices
- Technical indicator overlays
- Prediction accuracy comparisons
- Volume analysis

### Sink Validation

Run local unit/static checks without Elasticsearch:

```bash
python3 -m py_compile src/flink/es_sink.py tests/test_es_sink.py tests/test_es_sink_integration.py scripts/validate_es_sink.py scripts/benchmark_es_sink.py
python3 -m pytest -q tests/test_es_sink.py
```

Run the real Elasticsearch smoke test when the local cluster is available:

```bash
docker compose --profile elasticsearch up -d
python3 src/elasticsearch/init_indices.py
ES_INTEGRATION_TEST=1 python3 -m pytest -q tests/test_es_sink_integration.py
python3 scripts/validate_es_sink.py
python3 scripts/benchmark_es_sink.py --events 1000
```

For Kibana verification, create data views for `stock-engineered-features-*`
and `stock-predictions-*`, then filter for ticker `ESVALID` after running the
smoke validation script.
