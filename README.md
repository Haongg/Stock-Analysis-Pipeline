# Stock Analysis Pipeline

A production-ready Machine Learning pipeline for **real-time and batch stock price analysis and prediction**.

---

## Architecture

```
Stock-Analysis-Pipeline/
├── src/
│   ├── ingestion/          # Data acquisition (yfinance → PostgreSQL)
│   │   └── data_fetcher.py
│   ├── features/           # Technical indicator feature engineering
│   │   └── feature_engineering.py
│   ├── models/             # Model training with MLflow tracking
│   │   └── trainer.py
│   └── api/                # FastAPI inference endpoint
│       └── main.py
├── scripts/
│   └── train.py            # CLI entry-point for model training
├── tests/                  # Pytest unit tests
├── Dockerfile              # Multi-stage optimised image
├── docker-compose.yml      # App + PostgreSQL + MLflow
├── requirements.txt
├── .env.template
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Data Source | Yahoo Finance (`yfinance`) |
| Processing | Pandas, NumPy |
| ML | Scikit-learn (Random Forest), XGBoost |
| Tracking | MLflow (experiment tracking + model registry) |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL via SQLAlchemy |
| Container | Docker (multi-stage) + Docker Compose |

---

## Features Engineered

- Simple Moving Averages: SMA 10, SMA 20, SMA 50
- Exponential Moving Average: EMA 20
- Relative Strength Index (RSI 14)
- MACD + Signal line + Histogram
- Rolling Volatility (20-day)
- Lag features: `close_lag_1`, `close_lag_5`
- Daily return

---

## Quick Start

### 1. Copy environment template

```bash
cp .env.template .env
# Edit .env and set POSTGRES_PASSWORD and other values
```

### 2. Start all services

```bash
docker compose up -d
```

This starts:
- **PostgreSQL** on port `5432`
- **MLflow server** on port `5000`  →  http://localhost:5000
- **FastAPI app** on port `8000`    →  http://localhost:8000/docs

### 3. Train the model

```bash
docker compose run --rm trainer
```

Or locally (requires dependencies installed):

```bash
pip install -r requirements.txt
python scripts/train.py --tickers AAPL MSFT GOOG --start 2020-01-01
```

### 4. Request a prediction

```bash
# By ticker (fetches live data)
curl http://localhost:8000/predict/AAPL

# By raw feature vector
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"open":150,"high":155,"low":148,"volume":1000000,"sma_10":149,"sma_20":148,"sma_50":147,"ema_20":149,"rsi":55,"macd":1.2,"macd_signal":1.0,"macd_hist":0.2,"volatility":0.015,"close_lag_1":149,"close_lag_5":145,"daily_return":0.005}'
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Design Principles

- **Single Responsibility**: Each module owns exactly one concern (fetch / transform / train / serve).
- **Open/Closed**: New technical indicators can be added to `FeatureEngineer` without modifying existing methods.
- **Dependency Inversion**: `StockModelTrainer` and the API depend on DataFrames (abstractions), not concrete data-source implementations.
- **Interface Segregation**: `PredictionService` encapsulates model-loading logic so the API layer stays thin.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/predict/{ticker}` | Predict next-day close for a ticker |
| `POST` | `/predict` | Predict from a raw feature payload |

Interactive docs: http://localhost:8000/docs