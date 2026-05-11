# Stock Analysis Pipeline

A production-ready Machine Learning pipeline for **real-time streaming and batch stock price analysis and prediction** using Kafka, Elasticsearch, and modern ML stack.

---

## Architecture

```
Stock-Analysis-Pipeline/
├── src/
│   ├── ingestion/             # Data acquisition (yfinance)
│   │   └── data_fetcher.py
│   ├── streaming/             # Kafka producer for streaming data
│   │   └── kafka_producer.py
│   ├── elasticsearch/         # ES indices and mappings  
│   │   ├── init_indices.py
│   │   ├── stock_raw_ohlcv_mapping.json
│   │   ├── stock_engineered_features_mapping.json
│   │   └── stock_predictions_mapping.json
│   ├── features/              # Technical indicator feature engineering
│   │   └── feature_engineering.py
│   ├── models/                # Model training with MLflow tracking
│   │   └── trainer.py
|   ├── flink/                # Flink Jobs
│   │   └── main.py
│   └── api/                   # FastAPI inference endpoint
│       └── main.py
├── scripts/
│   ├── ingest_to_kafka.py     # Batch ingestion to Kafka
│   ├── ingest_to_kafka_streaming.py  # Real-time streaming ingestion
│   └── train.py               # CLI entry-point for model training
├── tests/                     # Pytest unit tests
├── Dockerfile                 # Multi-stage optimised image
├── docker-compose.yml         # Full stack orchestration
├── requirements.txt
├── .env.template
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Data Source | Yahoo Finance (`yfinance`) |
| **Message Broker** | **Apache Kafka (3-broker KRaft cluster)** |
| **Search & Analytics** | **Elasticsearch 8.10 + Kibana** |
| Processing | Pandas, NumPy |
| ML | Scikit-learn (Random Forest), XGBoost |
| Tracking | MLflow (experiment tracking + model registry) |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL via SQLAlchemy |
| Container | Docker (multi-stage) + Docker Compose |

---

## Features Engineered

- Simple Moving Averages: SMA 10, SMA 20, SMA 50
- Exponential Moving Average: EMA 20
- Relative Strength Index (RSI 14)
- MACD + Signal line + Histogram
- Rolling Volatility (20-day)
- Lag features: `close_lag_1`, `close_lag_5`
- Daily return

---

## Quick Start

### 1. Copy environment template

```bash
cp .env.template .env
# Edit .env and set POSTGRES_PASSWORD, KAFKA_BOOTSTRAP_SERVERS, ES_HOST, etc.
```

### 2. Start core services (PostgreSQL, MLflow, FastAPI)

```bash
docker compose --profile core up -d
```

This starts:
- **PostgreSQL** on port `5432`
- **MLflow server** on port `5000`  →  http://localhost:5000
- **FastAPI app** on port `8000`    →  http://localhost:8000/docs

**[Tips:] To optimize resource usage, start the core service independently unless other services are required**

### 3. Start Kafka cluster + Topic initialization

```bash
docker compose --profile kafka up -d
```

This creates:
- **Kafka brokers** on ports `19092, 29092, 39092`
- **Topic** `stock.raw.ohlcv` with 3 replicas

### 4. Initialize Elasticsearch + Kibana

```bash
docker compose --profile elasticsearch up -d
```

This creates:
- **Elasticsearch** on port `9200`
- **Kibana** on port `5601` → http://localhost:5601

### 5. Start Flink

```bash
docker compose --profile flink up -d
```

This creates:
- **Flink** on ports `8081`

> [!IMPORTANT:]
> **Put all python files into src\flink\ and trained models (.plk, .h5,...) in to models\ (if needed)** 
> **To ensure those files pushed into Flink server**
> **You can change the directories by modifying volumes in stock-jobmanager and stock-taskmanager (docker-compose.yml)**

### 6. Start data streaming (Yahoo Finance → Kafka)

```bash
# Option A: One-time batch ingestion
docker compose --profile kafka --profile ingestion up -d 

# Option B: Continuous real-time streaming (every 5 minutes) (suggested)
docker compose --profile kafka --profile streaming up -d
```
Or locally (requires dependencies installed) to save hardware resources:

```bash
# Option A: One-time batch ingestion
pip install -r requirements.txt
python scripts/ingest_to_kafka.py --tickers AAPL MSFT --start 2025-01-01

# Option B: Continuous real-time streaming (every 5 minutes) (suggested)
pip install -r requirements.txt
python scripts/ingest_to_kafka_streaming.py --tickers AAPL MSFT --interval 300
```

### 7. Initialize Elasticsearch Indices with Defined Mappings

```bash
docker compose --profile elasticsearch --profile indices up -d
```

Or locally (requires dependencies installed) to save hardware resources:

```bash
pip install -r requirements.txt
python src/elasticsearch/init_indices.py
```

This creates:
- **Elasticsearch** 3 indices:
  - `stock-raw-ohlcv`
  - `stock-engineered-features`
  - `stock-predictions`

> [!IMPORTANT:]
> **Dependency Warning: Step 7 relies on custom images generated in Step 6** 
> **To ensure proper resource optimization, complete Step 6 before proceeding to Step 7**

### 8. Train the model

```bash
docker compose run --rm trainer
```

Or locally (requires dependencies installed) to save hardware resources:

```bash
pip install -r requirements.txt
python scripts/train.py --tickers AAPL MSFT GOOG --start 2020-01-01
```

### 9. Request a prediction

```bash
# By ticker (fetches live data)
curl http://localhost:8000/predict/AAPL

# By raw feature vector
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"open":150,"high":155,"low":148,"volume":1000000,"sma_10":149,"sma_20":148,"sma_50":147,"ema_20":149,"rsi":55,"macd":1.2,"macd_signal":1.0,"macd_hist":0.2,"volatility":0.015,"close_lag_1":149,"close_lag_5":145,"daily_return":0.005}'
```

---

## Hardware Requirements

*You should have atleast 16GB RAM and 20GB Disk Space where images and containers are stored (ex: C://)*

## Monitoring & Checking Data

### Kafka Stream Data

Check if data is being streamed to Kafka:

```bash
# List all topics
docker compose exec kafka-1 /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka-1:9092 --list

# Describe topic details
docker compose exec kafka-1 /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka-1:9092 --describe --topic stock.raw.ohlcv

# Consume messages from the beginning (last 10 messages)
docker compose exec kafka-1 /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server kafka-1:9092 \
  --topic stock.raw.ohlcv \
  --from-beginning \
  --max-messages 10

# Consume new messages in real-time
docker compose exec kafka-1 /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server kafka-1:9092 \
  --topic stock.raw.ohlcv
```

### Elasticsearch Indices

Check Elasticsearch indices and data:

```bash
# List all indices with stats
curl http://localhost:9200/_cat/indices?v

# Get index statistics
curl http://localhost:9200/stock-raw-ohlcv/_stats

# Get index mapping
curl http://localhost:9200/stock-raw-ohlcv/_mapping?pretty

# Search data in index (retrieve first 10 documents)
curl -X GET "http://localhost:9200/stock-raw-ohlcv/_search?size=10&pretty"

# Search with query (e.g., specific ticker)
curl -X GET "http://localhost:9200/stock-raw-ohlcv/_search?pretty" -H 'Content-Type: application/json' \
  -d '{"query": {"match": {"ticker": "AAPL"}}}'
```

Or use **Kibana UI** for visual monitoring:
- URL: http://localhost:5601
- Create index patterns for each ES index
- Build dashboards to visualize stock data

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Design Principles

- **Single Responsibility**: Each module owns exactly one concern (fetch / transform / train / serve).
- **Open/Closed**: New technical indicators can be added to `FeatureEngineer` without modifying existing methods.
- **Dependency Inversion**: `StockModelTrainer` and the API depend on DataFrames (abstractions), not concrete data-source implementations.
- **Interface Segregation**: `PredictionService` encapsulates model-loading logic so the API layer stays thin.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/predict/{ticker}` | Predict next-day close for a ticker |
| `POST` | `/predict` | Predict from a raw feature payload |

Interactive docs: http://localhost:8000/docs