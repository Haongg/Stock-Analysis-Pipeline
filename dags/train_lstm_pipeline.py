import json
import logging
import re
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

from tasks.es_data_extractor import ESDataExtractor
from tasks.dataset_creator import create_dataset_task_from_parquet_paths


default_args = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

logger = logging.getLogger(__name__)


def _safe_version(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _build_flink_url(base_url: str, endpoint: str) -> str:
    normalized_base = base_url.rstrip("/") + "/"
    normalized_endpoint = endpoint.lstrip("/")
    return urljoin(normalized_base, normalized_endpoint)


def extract_features_task(**context):
    params = context.get("params", {})
    tickers = params.get("tickers") or ["AAPL", "MSFT", "GOOG"]
    days = int(params.get("lookback_days", 1095))

    extractor = ESDataExtractor()
    artifact_paths = []

    for ticker in tickers:
        df = extractor.extract_features(ticker=ticker, days=days)
        if not extractor.validate_data(df):
            raise ValueError(f"Data validation failed for ticker={ticker}")
        path = extractor.save_parquet(df=df, ticker=ticker)
        artifact_paths.append(path)

    return artifact_paths


def create_dataset_task(**context):
    params = context.get("params", {})
    seq_length = int(params.get("seq_length", 30))
    output_dir = params.get("dataset_output_dir", "data")

    ti = context["ti"]
    parquet_paths = ti.xcom_pull(task_ids="extract_features")
    if not parquet_paths:
        raise ValueError("No parquet paths returned from extract_features task.")

    return create_dataset_task_from_parquet_paths(
        parquet_paths=parquet_paths,
        seq_length=seq_length,
        output_dir=output_dir,
    )


def notify_flink_reload_task(**context):
    params = context.get("params", {})
    ti = context["ti"]
    dag_run = context.get("dag_run")
    task = context.get("task")

    promoted_version = ti.xcom_pull(task_ids="promote_model", key="model_version")
    run_id = dag_run.run_id if dag_run else context.get("run_id", "manual")
    version = promoted_version or f"manual__{_safe_version(run_id)}"

    conn = BaseHook.get_connection("flink_default")
    endpoint = params.get("flink_reload_endpoint", "/flink/model-reload")
    url = _build_flink_url(conn.get_uri(), endpoint)
    payload = {
        "version": version,
        "model_name": params.get("model_name", "lstm_stock_predictor"),
        "stage": params.get("model_stage", "production"),
        "dag_id": context["dag"].dag_id,
        "run_id": run_id,
        "task_id": task.task_id if task else "notify_flink_reload",
    }

    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=int(params.get("flink_reload_timeout_seconds", 30))) as response:
            status = response.status
            body = response.read(1000).decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"Flink reload failed with HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Flink reload request failed: {exc.reason}") from exc

    if status < 200 or status >= 300:
        raise RuntimeError(f"Flink reload failed with HTTP {status}: {body}")

    logger.info("Flink model reload accepted: status=%s body=%s payload=%s", status, body, payload)
    return {"status": status, "version": version, "response": body}


with DAG(
    dag_id="train_lstm_pipeline",
    default_args=default_args,
    description="Daily LSTM retraining pipeline skeleton",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    params={
        "flink_reload_endpoint": "/flink/model-reload",
        "model_name": "lstm_stock_predictor",
        "model_stage": "production",
        "flink_reload_timeout_seconds": 30,
    },
    tags=["lstm", "daily-retraining", "skeleton"],
) as dag:
    extract_features = PythonOperator(
        task_id="extract_features",
        python_callable=extract_features_task,
    )
    create_dataset = PythonOperator(
        task_id="create_dataset",
        python_callable=create_dataset_task,
    )
    train_lstm = EmptyOperator(task_id="train_lstm")
    evaluate = EmptyOperator(task_id="evaluate")
    convert_onnx = EmptyOperator(task_id="convert_onnx")
    register_mlflow = EmptyOperator(task_id="register_mlflow")
    promote_model = EmptyOperator(task_id="promote_model")
    notify_flink_reload = PythonOperator(
        task_id="notify_flink_reload",
        python_callable=notify_flink_reload_task,
    )

    (
        extract_features
        >> create_dataset
        >> train_lstm
        >> evaluate
        >> convert_onnx
        >> register_mlflow
        >> promote_model
        >> notify_flink_reload
    )
