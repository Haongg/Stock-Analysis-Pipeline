# Flink Application Mode Runbook

This runbook validates the Flink Application Mode deployment used by the stock analytics pipeline. It is intentionally scoped to Flink JobManager, TaskManager, and submitted-job visibility. The broader Docker startup checklist is tracked separately in `docs/end_to_end_startup_checklist.md`.

## Topology

- `stock-jobmanager` runs Flink Application Mode with:
  ```bash
  standalone-job --python /opt/flink/usrlib/python_code/main.py
  ```
- `stock-taskmanager` connects to `stock-jobmanager`.
- This Application Mode cluster is intended for one Flink job. Create another cluster if a second independent Flink job is needed.
- Source code is mounted from `./src/flink` to `/opt/flink/usrlib/python_code`.
- Model artifacts are mounted from `./models` to `/opt/flink/usrlib/models`.

## Prerequisites

- Docker is running.
- Kafka profile is running and topic `stock.raw.ohlcv` exists.
- `src/flink/main.py` is present and importable by the Flink image.
- LSTM model/scaler artifacts are optional for Application Mode validation. Until tasks `2.4-2.9` create real artifacts, prediction records may contain `prediction_error="model_not_loaded"`.

## Build and Start

Build the Flink image:

```bash
docker compose build stock-jobmanager stock-taskmanager
```

Start Kafka and Flink:

```bash
docker compose --profile kafka --profile flink up -d
```

Kafka is included here because the application job reads from `stock.raw.ohlcv`.

## Validation Commands

Check containers:

```bash
docker compose ps stock-jobmanager stock-taskmanager
```

Check Flink Web UI endpoints:

```bash
curl -f http://localhost:8081/overview
curl -f http://localhost:8081/taskmanagers
curl -f http://localhost:8081/jobs/overview
```

Check submitted job visibility:

```bash
python3 scripts/check_e2e_stack.py --skip-kafka --require-flink-job
```

Inspect JobManager logs:

```bash
docker compose logs --tail=100 stock-jobmanager
```

Open the UI:

```bash
open http://localhost:8081
```

Expected result:

- JobManager endpoint responds.
- At least one TaskManager is visible.
- A submitted job is visible in `/jobs/overview`.
- A job in `RUNNING`, `CREATED`, `RESTARTING`, `INITIALIZING`, or another non-terminal active state passes the `--require-flink-job` smoke check.

## Troubleshooting

- **No Kafka topic**: run `docker compose --profile kafka up -d` and wait for `kafka-init`.
- **No TaskManager visible**: inspect `docker compose logs --tail=100 stock-taskmanager` and confirm `jobmanager.rpc.address: stock-jobmanager`.
- **No submitted job visible**: inspect `stock-jobmanager` logs for PyFlink import errors or startup failures.
- **PyFlink dependency errors**: rebuild with `docker compose build stock-jobmanager stock-taskmanager`.
- **Missing model/scaler artifacts**: this should not block Application Mode validation; it may only make inference output use `model_not_loaded`.
- **Job failed/canceled**: inspect logs and rerun the smoke check after fixing the startup error.
