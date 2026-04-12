"""
scripts/ingest_to_kafka_streaming.py

Streaming ingestion service that continuously fetches latest stock data from Yahoo Finance
and publishes to Kafka. Runs on a schedule to avoid data duplication.

Usage
-----
    python scripts/ingest_to_kafka_streaming.py --tickers AAPL MSFT --interval 300
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, UTC, timedelta
from typing import Any, Dict, Optional, Set

import pandas as pd
import schedule

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.data_fetcher import StockDataFetcher
from src.streaming.kafka_producer import close_producer, create_kafka_producer, publish_json


class StreamingIngestor:
    """Handles streaming ingestion with duplicate prevention."""

    def __init__(self, tickers: list[str], interval_seconds: int = 300):
        self.tickers = [t.upper() for t in tickers]
        self.interval_seconds = interval_seconds
        self.fetcher = StockDataFetcher()
        self.producer = create_kafka_producer()

        # Track last ingested date for each ticker to prevent duplicates
        self.last_ingested_dates: Dict[str, pd.Timestamp] = {}

        # Track processed records to avoid duplicates within same run
        self.processed_records: Set[str] = set()

        print(f"[INIT] Streaming ingestor started for tickers: {self.tickers}")
        print(f"[INIT] Fetch interval: {self.interval_seconds} seconds")

    def _generate_record_key(self, ticker: str, date: str) -> str:
        """Generate unique key for record to prevent duplicates."""
        return f"{ticker}_{date}"

    def _should_process_record(self, ticker: str, date: pd.Timestamp) -> bool:
        """Check if record should be processed (not duplicate)."""
        record_key = self._generate_record_key(ticker, date.strftime("%Y-%m-%d"))

        # Check if already processed in this session
        if record_key in self.processed_records:
            return False

        # Check if newer than last ingested date for this ticker
        last_date = self.last_ingested_dates.get(ticker)
        if last_date and date <= last_date:
            return False

        return True

    def _update_last_ingested_date(self, ticker: str, date: pd.Timestamp) -> None:
        """Update the last ingested date for a ticker."""
        current_last = self.last_ingested_dates.get(ticker)
        if current_last is None or date > current_last:
            self.last_ingested_dates[ticker] = date

    def _to_event(self, index_dt: pd.Timestamp, row: pd.Series) -> Dict[str, Any]:
        """Convert pandas row to event dict."""
        # event_date = index_dt.to_pydatetime().replace(tzinfo=pd.Timestamp.utcnow().tz).isoformat().replace("+00:00", "Z")
        # ingested_at = datetime.utcnow().replace(tzinfo=pd.UTC).isoformat().replace("+00:00", "Z")

        event_date = (
            index_dt.tz_localize(UTC) if index_dt.tzinfo is None else index_dt
        ).isoformat().replace("+00:00", "Z")

        ingested_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

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

    def ingest_latest_data(self) -> None:
        """Fetch latest data for all tickers and publish new records to Kafka."""
        print(f"\n[{datetime.now()}] Starting ingestion cycle...")

        total_sent = 0
        total_skipped = 0

        try:
            for ticker in self.tickers:
                try:
                    # Fetch latest 7 days to ensure we get recent data
                    # This covers weekends and ensures we don't miss data
                    df = self.fetcher.fetch_latest(ticker=ticker, lookback_days=7)

                    if df.empty:
                        print(f"[WARN] No data fetched for {ticker}")
                        continue

                    ticker_sent = 0
                    ticker_skipped = 0

                    # Process each row, checking for duplicates
                    for idx, row in df.iterrows():
                        if self._should_process_record(ticker, idx):
                            event = self._to_event(idx, row)
                            try:
                                publish_json(producer=self.producer, key=event["ticker"], value=event)
                                self.processed_records.add(self._generate_record_key(ticker, idx.strftime("%Y-%m-%d")))
                                self._update_last_ingested_date(ticker, idx)
                                ticker_sent += 1
                            except Exception as exc:
                                print(f"[ERROR] Failed to publish {ticker} at {idx.strftime('%Y-%m-%d')}: {exc}")
                        else:
                            ticker_skipped += 1

                    if ticker_sent > 0:
                        print(f"[INFO] {ticker}: sent {ticker_sent} records, skipped {ticker_skipped} duplicates")
                    else:
                        print(f"[INFO] {ticker}: no new records (skipped {ticker_skipped})")

                    total_sent += ticker_sent
                    total_skipped += ticker_skipped

                except Exception as exc:
                    print(f"[ERROR] Failed to process {ticker}: {exc}")

            print(f"[DONE] Cycle completed - sent: {total_sent}, skipped: {total_skipped}")

        except Exception as exc:
            print(f"[ERROR] Ingestion cycle failed: {exc}")

    def run(self) -> None:
        """Run the streaming ingestion service."""
        # Schedule the ingestion
        schedule.every(self.interval_seconds).seconds.do(self.ingest_latest_data)

        # Run initial ingestion immediately
        print("[START] Running initial ingestion...")
        self.ingest_latest_data()

        print(f"[START] Scheduled ingestion every {self.interval_seconds} seconds. Press Ctrl+C to stop.")

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[STOP] Shutting down streaming ingestor...")
        finally:
            close_producer(self.producer)
            print("[STOP] Streaming ingestor stopped.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Streaming ingestion from Yahoo Finance to Kafka")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL"],
        help="Ticker symbols to stream (e.g. AAPL MSFT GOOG)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Ingestion interval in seconds (default: 300 = 5 minutes)"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    ingestor = StreamingIngestor(
        tickers=args.tickers,
        interval_seconds=args.interval
    )
    ingestor.run()


if __name__ == "__main__":
    main()