#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_TOPIC = "stock.raw.ohlcv"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def http_get(url: str, timeout: float = 5.0) -> tuple[bool, str, Any | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "stock-e2e-check/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(4096)
            status = response.getcode()
            text = body.decode("utf-8", errors="replace")
            parsed = None
            if text:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = text
            return 200 <= status < 400, f"HTTP {status}", parsed
    except urllib.error.HTTPError as exc:
        return exc.code in {401, 403}, f"HTTP {exc.code}", None
    except Exception as exc:
        return False, str(exc), None


def run_command(command: list[str], timeout: float = 15.0) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return False, str(exc)

    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode == 0:
        return True, output or "ok"
    return False, output or f"exit code {completed.returncode}"


def check_http(name: str, url: str, required: bool = True) -> CheckResult:
    ok, detail, parsed = http_get(url)
    if isinstance(parsed, dict) and "status" in parsed:
        detail = f"{detail}, status={parsed['status']}"
    return CheckResult(name=name, ok=ok, detail=f"{url} -> {detail}", required=required)


def check_flink_job(flink_url: str) -> CheckResult:
    url = f"{flink_url.rstrip('/')}/jobs/overview"
    ok, detail, parsed = http_get(url)
    if not ok:
        return CheckResult("Flink submitted job", False, f"{url} -> {detail}")

    jobs = []
    if isinstance(parsed, dict):
        jobs = parsed.get("jobs") or []
    if not jobs:
        return CheckResult("Flink submitted job", False, "no submitted jobs visible")

    terminal_states = {"FAILED", "CANCELED", "CANCELLED", "FINISHED"}
    active_jobs = []
    summaries = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        job_id = job.get("jid") or job.get("id") or "unknown"
        name = job.get("name") or "unknown"
        state = str(job.get("state") or "UNKNOWN").upper()
        summaries.append(f"{job_id}:{name}:{state}")
        if state not in terminal_states:
            active_jobs.append(job)

    if not active_jobs:
        return CheckResult("Flink submitted job", False, "only terminal jobs visible: " + ", ".join(summaries))
    return CheckResult("Flink submitted job", True, "active jobs: " + ", ".join(summaries))


def check_kafka_topic(topic: str = DEFAULT_TOPIC) -> CheckResult:
    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        "kafka-1",
        "/opt/kafka/bin/kafka-topics.sh",
        "--bootstrap-server",
        "kafka-1:9092",
        "--describe",
        "--topic",
        topic,
    ]
    ok, detail = run_command(command)
    if ok and topic not in detail:
        ok = False
        detail = f"topic {topic} not found in describe output"
    return CheckResult("Kafka topic", ok, detail)


def check_sample_ticker(ticker: str, es_url: str) -> CheckResult:
    query = urllib.parse.quote(ticker)
    url = f"{es_url.rstrip('/')}/stock-*/_search?q=ticker:{query}&size=1"
    ok, detail, parsed = http_get(url)
    if not ok:
        return CheckResult("Sample ticker search", False, f"{url} -> {detail}", required=False)

    hits = 0
    if isinstance(parsed, dict):
        total = parsed.get("hits", {}).get("total", {})
        if isinstance(total, dict):
            hits = int(total.get("value", 0))
        elif isinstance(total, int):
            hits = total
    status = "found" if hits > 0 else "no matching documents yet"
    return CheckResult("Sample ticker search", True, f"{ticker}: {status} ({hits} hits)", required=False)


def print_result(result: CheckResult) -> None:
    marker = "PASS" if result.ok else ("WARN" if not result.required else "FAIL")
    print(f"[{marker}] {result.name}: {result.detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only smoke checks for the stock analytics Docker stack.")
    parser.add_argument("--include-airflow", action="store_true", help="Check Airflow webserver on localhost:8080.")
    parser.add_argument("--include-core", action="store_true", help="Check MLflow and FastAPI core services.")
    parser.add_argument("--include-kibana", action="store_true", help="Check Kibana on localhost:5601.")
    parser.add_argument("--sample-ticker", default=None, help="Optionally query Elasticsearch stock-* indices for a ticker.")
    parser.add_argument("--es-url", default="http://localhost:9200", help="Elasticsearch base URL.")
    parser.add_argument("--flink-url", default="http://localhost:8081", help="Flink Web UI base URL.")
    parser.add_argument("--mlflow-url", default="http://localhost:5000", help="MLflow base URL.")
    parser.add_argument("--app-url", default="http://localhost:8000", help="FastAPI app base URL.")
    parser.add_argument("--kibana-url", default="http://localhost:5601", help="Kibana base URL.")
    parser.add_argument("--airflow-url", default="http://localhost:8080", help="Airflow webserver base URL.")
    parser.add_argument("--kafka-topic", default=DEFAULT_TOPIC, help="Kafka topic to describe.")
    parser.add_argument("--skip-kafka", action="store_true", help="Skip docker compose Kafka topic check.")
    parser.add_argument("--require-flink-job", action="store_true", help="Fail if no active Flink job is visible.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results: list[CheckResult] = []

    results.append(check_http("Elasticsearch cluster", f"{args.es_url.rstrip('/')}/_cluster/health"))
    results.append(check_http("Flink overview", f"{args.flink_url.rstrip('/')}/overview"))
    results.append(check_http("Flink jobs", f"{args.flink_url.rstrip('/')}/jobs/overview"))
    if args.require_flink_job:
        results.append(check_flink_job(args.flink_url))

    if not args.skip_kafka:
        results.append(check_kafka_topic(args.kafka_topic))

    if args.include_core:
        results.append(check_http("MLflow", args.mlflow_url.rstrip("/")))
        results.append(check_http("FastAPI health", f"{args.app_url.rstrip('/')}/health"))

    if args.include_kibana:
        results.append(check_http("Kibana", args.kibana_url.rstrip("/")))

    if args.include_airflow:
        results.append(check_http("Airflow", args.airflow_url.rstrip("/")))

    if args.sample_ticker:
        results.append(check_sample_ticker(args.sample_ticker, args.es_url))

    for result in results:
        print_result(result)

    failed_required = [result for result in results if result.required and not result.ok]
    if failed_required:
        print(f"\n{len(failed_required)} required check(s) failed.", file=sys.stderr)
        return 1

    print("\nAll required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
