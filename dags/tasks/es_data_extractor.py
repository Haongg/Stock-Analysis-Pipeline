from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from airflow.hooks.base import BaseHook
from elasticsearch import Elasticsearch


logger = logging.getLogger(__name__)


class ESDataExtractor:
    REQUIRED_COLUMNS = [
        "@timestamp",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_20",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "volatility",
        "daily_return",
        "close_lag_1",
        "close_lag_5",
    ]

    def __init__(
        self,
        conn_id: str = "elasticsearch_default",
        index_name: str = "stock-engineered-features",
        page_size: int = 1000,
        min_records: int = 120,
        output_dir: str = "data",
    ) -> None:
        self.conn_id = conn_id
        self.index_name = index_name
        self.page_size = page_size
        self.min_records = min_records
        self.output_dir = output_dir

    def _create_client(self) -> Elasticsearch:
        conn = BaseHook.get_connection(self.conn_id)
        host = conn.host or "localhost"
        port = conn.port or 9200
        scheme = conn.schema or "http"
        if conn.login and conn.password:
            return Elasticsearch(
                f"{scheme}://{host}:{port}",
                basic_auth=(conn.login, conn.password),
                request_timeout=60,
            )
        return Elasticsearch(f"{scheme}://{host}:{port}", request_timeout=60)

    def _build_query(self, ticker: str, days: int) -> dict[str, Any]:
        return {
            "size": self.page_size,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"ticker": ticker.upper()}},
                        {"range": {"@timestamp": {"gte": f"now-{days}d/d", "lte": "now/d"}}},
                    ]
                }
            },
            "sort": [
                {"@timestamp": "asc"},
                {"_id": "asc"},
            ],
            "_source": self.REQUIRED_COLUMNS,
        }

    def extract_features(self, ticker: str, days: int = 1095) -> pd.DataFrame:
        client = self._create_client()
        query = self._build_query(ticker=ticker, days=days)

        all_hits: list[dict[str, Any]] = []
        response = client.search(index=self.index_name, body=query)
        hits = response.get("hits", {}).get("hits", [])
        all_hits.extend(hits)

        while hits:
            last_sort = hits[-1].get("sort")
            if not last_sort:
                break
            query["search_after"] = last_sort
            response = client.search(index=self.index_name, body=query)
            hits = response.get("hits", {}).get("hits", [])
            all_hits.extend(hits)

        rows = [hit.get("_source", {}) for hit in all_hits]
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["@timestamp"] = pd.to_datetime(df["@timestamp"], utc=True, errors="coerce")
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df.sort_values(["@timestamp"]).drop_duplicates(subset=["ticker", "@timestamp"], keep="last")
        df = df.reset_index(drop=True)
        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        if df.empty:
            logger.warning("ESDataExtractor validation failed: dataframe is empty")
            return False

        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            logger.warning("ESDataExtractor validation failed: missing columns %s", missing_cols)
            return False

        if len(df) < self.min_records:
            logger.warning(
                "ESDataExtractor validation failed: only %s records, need >= %s",
                len(df),
                self.min_records,
            )
            return False

        if df["@timestamp"].isna().any():
            logger.warning("ESDataExtractor validation failed: invalid @timestamp values")
            return False

        must_not_null = ["@timestamp", "ticker", "close", "volume"]
        for col in must_not_null:
            null_rate = float(df[col].isna().mean())
            if null_rate > 0.01:
                logger.warning("ESDataExtractor validation failed: null rate too high for %s = %.3f", col, null_rate)
                return False

        if not df["@timestamp"].is_monotonic_increasing:
            logger.warning("ESDataExtractor validation failed: @timestamp is not monotonic increasing")
            return False

        return True

    def save_parquet(self, df: pd.DataFrame, ticker: str, run_date: str | None = None) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        date_part = run_date or datetime.now(timezone.utc).strftime("%Y%m%d")
        output_path = os.path.join(self.output_dir, f"features_{ticker.upper()}_{date_part}.parquet")
        df.to_parquet(output_path, index=False)
        return output_path
