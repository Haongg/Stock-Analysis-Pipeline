"""
Entry-point script for training the stock prediction model.

Usage:
    python scripts/train.py --tickers AAPL MSFT GOOG \
        --start 2020-01-01 --end 2024-01-01 \
        --n-estimators 200 --max-depth 15
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sqlalchemy import create_engine

from src.features.feature_engineering import FeatureEngineer
from src.ingestion.data_fetcher import StockDataFetcher
from src.models.trainer import StockModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stock price prediction model")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL"],
        help="List of ticker symbols to include in training data",
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--experiment", default="stock_analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Optionally persist raw data to Postgres
    db_url = os.getenv("DATABASE_URL")
    engine = create_engine(db_url) if db_url else None

    fetcher = StockDataFetcher(engine=engine)
    engineer = FeatureEngineer()

    all_frames: list[pd.DataFrame] = []
    for ticker in args.tickers:
        logger.info("Processing ticker: %s", ticker)
        raw = fetcher.fetch_historical(ticker, start=args.start, end=args.end)
        if raw.empty:
            logger.warning("Skipping %s – no data returned", ticker)
            continue
        if engine:
            fetcher.save_to_db(raw)
        features = engineer.transform(raw)
        all_frames.append(features)

    if not all_frames:
        logger.error("No data available for training.  Exiting.")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=False).sort_index()
    logger.info("Total training rows: %d", len(combined))

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": 42,
    }

    trainer = StockModelTrainer(
        params=params,
        experiment_name=args.experiment,
    )
    _, metrics = trainer.train(combined)

    logger.info("Final metrics – RMSE: %.4f  MAE: %.4f  R²: %.4f",
                metrics["rmse"], metrics["mae"], metrics["r2"])


if __name__ == "__main__":
    main()
