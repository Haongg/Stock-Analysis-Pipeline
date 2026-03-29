"""
FastAPI inference endpoint for real-time stock price prediction.

Exposes:
  GET  /health              – liveness probe
  GET  /predict/{ticker}   – fetch recent data, engineer features, return prediction
  POST /predict             – accept raw feature payload and return prediction

Follows Interface Segregation Principle: prediction logic is encapsulated in a
dedicated PredictionService so that the API layer stays thin.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.features.feature_engineering import FeatureEngineer
from src.ingestion.data_fetcher import StockDataFetcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class FeaturePayload(BaseModel):
    """Raw feature vector for a single prediction request."""

    open: float
    high: float
    low: float
    volume: float
    sma_10: float
    sma_20: float
    sma_50: float
    ema_20: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    volatility: float
    close_lag_1: float
    close_lag_5: float
    daily_return: float


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    ticker: Optional[str] = None
    predicted_close: float
    model_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Prediction service
# ---------------------------------------------------------------------------


class PredictionService:
    """Loads an MLflow model and serves predictions."""

    def __init__(self) -> None:
        self._model = None
        self._model_version: Optional[str] = None

    def load_model(self) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        model_name = os.getenv("MODEL_NAME", "stock_price_predictor")
        model_stage = os.getenv("MODEL_STAGE", "latest")

        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{model_name}/{model_stage}"

        logger.info("Loading model from %s", model_uri)
        try:
            self._model = mlflow.sklearn.load_model(model_uri)
            self._model_version = model_stage
            logger.info("Model loaded successfully")
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            self._model = None

    def predict(self, features: pd.DataFrame) -> float:
        if self._model is None:
            raise RuntimeError("Model is not loaded")
        prediction = self._model.predict(features)
        return float(prediction[0])

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def model_version(self) -> Optional[str]:
        return self._model_version


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

_service = PredictionService()
_fetcher = StockDataFetcher()
_engineer = FeatureEngineer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _service.load_model()
    yield


app = FastAPI(
    title="Stock Analysis Prediction API",
    description="Real-time stock price prediction using a trained Random Forest model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
def health_check() -> Dict[str, str]:
    """Liveness probe."""
    status = "ready" if _service.is_ready else "model_not_loaded"
    return {"status": status}


@app.get(
    "/predict/{ticker}",
    response_model=PredictionResponse,
    tags=["prediction"],
)
def predict_ticker(
    ticker: str,
    lookback_days: int = Query(default=90, ge=60, le=365),
) -> PredictionResponse:
    """Fetch recent market data for *ticker*, compute features, and predict the
    next-day closing price.

    Parameters
    ----------
    ticker:
        Stock symbol (e.g. ``AAPL``).
    lookback_days:
        Number of calendar days of historical data to fetch (default: 90).
    """
    if not _service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = _fetcher.fetch_latest(ticker.upper(), lookback_days=lookback_days)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for ticker '{ticker}'",
        )

    df_features = _engineer.transform(df)
    if df_features.empty:
        raise HTTPException(
            status_code=422,
            detail="Not enough data to compute features",
        )

    feature_cols = [
        "open", "high", "low", "volume",
        "sma_10", "sma_20", "sma_50", "ema_20",
        "rsi", "macd", "macd_signal", "macd_hist",
        "volatility", "close_lag_1", "close_lag_5", "daily_return",
    ]
    latest = df_features[feature_cols].iloc[[-1]]
    prediction = _service.predict(latest)

    return PredictionResponse(
        ticker=ticker.upper(),
        predicted_close=round(prediction, 4),
        model_version=_service.model_version,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
)
def predict_features(payload: FeaturePayload) -> PredictionResponse:
    """Accept a raw feature vector and return the predicted closing price.

    Useful for batch clients that pre-compute features themselves.
    """
    if not _service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features_df = pd.DataFrame([payload.model_dump()])
    prediction = _service.predict(features_df)

    return PredictionResponse(
        predicted_close=round(prediction, 4),
        model_version=_service.model_version,
    )
