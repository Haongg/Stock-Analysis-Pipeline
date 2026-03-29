"""
scripts/train.py

Entry-point for model training.

Usage
-----
    python scripts/train.py --tickers AAPL MSFT --start 2020-01-01
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stock price prediction model")
    parser.add_argument("--tickers", nargs="+", default=["AAPL"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--experiment", default="stock_analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: wire up StockDataFetcher → FeatureEngineer → StockModelTrainer
    raise NotImplementedError


if __name__ == "__main__":
    main()
