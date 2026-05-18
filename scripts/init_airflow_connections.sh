#!/usr/bin/env bash
set -euo pipefail

add_or_update_connection() {
  local conn_id="$1"
  local conn_uri="$2"

  if airflow connections get "$conn_id" >/dev/null 2>&1; then
    airflow connections delete "$conn_id" >/dev/null
  fi
  airflow connections add "$conn_id" --conn-uri "$conn_uri" >/dev/null
  echo "[airflow-init] upserted connection: $conn_id"
}

add_or_update_connection "elasticsearch_default" "${AIRFLOW_CONN_ELASTICSEARCH_DEFAULT}"
add_or_update_connection "postgres_default" "${AIRFLOW_CONN_POSTGRES_DEFAULT}"
add_or_update_connection "mlflow_default" "${AIRFLOW_CONN_MLFLOW_DEFAULT}"
add_or_update_connection "flink_default" "${AIRFLOW_CONN_FLINK_DEFAULT}"

echo "[airflow-init] connections bootstrap completed"
