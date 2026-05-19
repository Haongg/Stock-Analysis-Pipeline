# End-to-End Docker Startup Checklist

This checklist starts the Docker-based stock analytics stack in a repeatable order and verifies the visible service boundaries. It is a smoke checklist, not a production load test.

## Prerequisites

- Docker Desktop or Docker Engine is running.
- `.env` exists:
  ```bash
  cp .env.template .env
  ```
- Required ports are available: `5432`, `5000`, `8000`, `19092`, `29092`, `39092`, `9200`, `5601`, `8081`, `8080`.
- Model artifacts are optional for startup validation. Until tasks `2.4-2.9` produce a real LSTM/ONNX artifact, Flink prediction records may emit `prediction_error="model_not_loaded"`.

## Startup Order

### 1. Core services: Postgres, MLflow, FastAPI

```bash
docker compose --profile core up -d
```

Validate:

```bash
docker compose ps
curl -f http://localhost:5000
curl -f http://localhost:8000/health
```

### 2. Kafka cluster and topic initialization

```bash
docker compose --profile kafka up -d
```

Validate:

```bash
docker compose exec kafka-1 /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka-1:9092 --list
docker compose exec kafka-1 /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka-1:9092 --describe --topic stock.raw.ohlcv
```

Expected topic: `stock.raw.ohlcv`.

### 3. Elasticsearch and Kibana

```bash
docker compose --profile elasticsearch up -d
```

Validate:

```bash
curl -f http://localhost:9200/_cluster/health
open http://localhost:5601
```

Kibana may take longer than Elasticsearch to become ready.

### 4. Elasticsearch index initialization

```bash
docker compose --profile elasticsearch --profile indices up -d
```

Validate:

```bash
curl -f "http://localhost:9200/_cat/indices/stock-*?v"
```

Expected logical index families:

- `stock-raw-ohlcv`
- `stock-engineered-features`
- `stock-predictions`

Feature and prediction writes may use date-suffixed physical indices such as `stock-predictions-2025-01-15`.

### 5. Flink Application Mode cluster

```bash
docker compose --profile flink up -d
```

Validate:

```bash
curl -f http://localhost:8081/overview
curl -f http://localhost:8081/jobs/overview
```

Open the Flink UI at http://localhost:8081 and confirm the JobManager and TaskManager are visible. Application Mode-specific job submission validation remains tracked by task `8.2`.

### 6. Data ingestion

Run one batch ingestion:

```bash
docker compose --profile kafka --profile ingestion up -d
```

Or start continuous streaming ingestion:

```bash
docker compose --profile kafka --profile streaming up -d
```

Validate sample ticker messages:

```bash
docker compose exec kafka-1 /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server kafka-1:9092 \
  --topic stock.raw.ohlcv \
  --from-beginning \
  --max-messages 5
```

### 7. Optional Airflow stack

Airflow is optional in the serving-path startup because LSTM training tasks `2.4-2.9` are still pending.

```bash
docker compose --profile airflow up -d
```

Validate:

```bash
open http://localhost:8080
```

Default local credentials are configured through `.env` / compose defaults.

## Automated Smoke Check

After starting the desired profiles, run:

```bash
python3 scripts/check_e2e_stack.py --include-core --include-kibana --sample-ticker AAPL
```

Include Airflow only when the Airflow profile is running:

```bash
python3 scripts/check_e2e_stack.py --include-core --include-kibana --include-airflow
```

The script is read-only. It checks HTTP endpoints, Kafka topic metadata, and optionally searches Elasticsearch for the sample ticker.

## Troubleshooting

- If Kafka topic checks fail, wait for `kafka-init` to complete and rerun the topic describe command.
- If Elasticsearch health is `yellow`, that can be expected on a single-node local cluster with replicas configured.
- If Flink emits `model_not_loaded`, provide real `FLINK_MODEL_PATH` and `FLINK_SCALER_PATH` artifacts after tasks `2.4-2.9` are implemented.
- If a port is already in use, stop the conflicting local service or adjust the port mapping in `docker-compose.yml`.
