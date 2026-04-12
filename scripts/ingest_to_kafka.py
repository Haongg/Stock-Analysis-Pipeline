"""
scripts/ingest_to_kafka.py

Entry-point for ingesting OHLCV data from Yahoo Finance to Kafka.

Usage
-----
    python scripts/ingest_to_kafka.py --tickers AAPL MSFT --start 2024-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.data_fetcher import StockDataFetcher
from src.streaming.kafka_producer import close_producer, create_kafka_producer, publish_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest OHLCV data from Yahoo Finance to Kafka")
    parser.add_argument("--tickers", nargs="+", default=["AAPL"], help="Ticker symbols, e.g. AAPL MSFT")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Optional end date (YYYY-MM-DD)")
    return parser.parse_args()


# Convert each pandas row into JSON-like dictionary
def _to_event(index_dt: pd.Timestamp, row: pd.Series) -> Dict[str, Any]:
    # Convert time from pandas format into python format, then convert into UTC and replace +00:00 with Z by ISO format
    event_date = index_dt.to_pydatetime().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    # Get current time with UTC and replace +00:00 with Z by ISO format
    ingested_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    # Conventional schema
    return {
        "ticker": str(row["ticker"]),
        "date": event_date,
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
        "ingested_at": ingested_at,
    }


def main() -> None:
    args = parse_args()
    fetcher = StockDataFetcher()
    producer = create_kafka_producer()
    sent_success = 0
    sent_failed = 0

    try:
        for ticker in args.tickers:
            df = fetcher.fetch_historical(ticker=ticker, start=args.start, end=args.end) # đang lấy dữ liệu historical
            if df.empty:
                print(f"[WARN] No rows fetched for {ticker}")
                continue

            for idx, row in df.iterrows():
                event = _to_event(idx, row)
                try:
                    publish_json(producer=producer, key=event["ticker"], value=event)
                    sent_success += 1
                except Exception as exc:
                    sent_failed += 1
                    print(
                        f"[ERROR] Failed to publish record for {event['ticker']} "
                        f"at {event['date']}: {exc}"
                    )

            print(f"[INFO] Published {len(df)} rows for {ticker}")
    finally:
        close_producer(producer)

    print(
        f"[DONE] Total published records: {sent_success} "
        f"(success={sent_success}, failed={sent_failed})"
    )


if __name__ == "__main__":
    main()
