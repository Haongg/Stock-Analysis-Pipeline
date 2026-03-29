"""
api/main.py

FastAPI inference service.

Routes
------
GET  /health              – liveness probe
GET  /predict/{ticker}   – fetch live data, compute features, return prediction
POST /predict             – accept a raw feature payload, return prediction
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Stock Analysis Prediction API",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class FeaturePayload(BaseModel):
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
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
def health_check():
    """Return service liveness status."""
    raise NotImplementedError


@app.get("/predict/{ticker}", response_model=PredictionResponse, tags=["prediction"])
def predict_ticker(ticker: str):
    """Fetch recent data for *ticker*, compute features, predict next-day close."""
    raise NotImplementedError


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_features(payload: FeaturePayload):
    """Predict next-day close from a raw feature vector."""
    raise NotImplementedError
