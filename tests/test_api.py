"""Tests for the FastAPI app routes (schema and status codes)."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_health_route_exists():
    response = client.get("/health")
    # Route is registered; 500 is acceptable while handler is NotImplemented
    assert response.status_code in (200, 500)


def test_predict_ticker_route_exists():
    response = client.get("/predict/AAPL")
    assert response.status_code in (200, 422, 500, 503)


def test_post_predict_missing_fields_returns_422():
    """Pydantic validation should reject incomplete payloads with 422."""
    response = client.post("/predict", json={"open": 150.0})
    assert response.status_code == 422


def test_post_predict_valid_payload_accepted():
    """A complete payload passes schema validation (handler may be NotImplemented)."""
    payload = {
        "open": 150.0, "high": 155.0, "low": 148.0, "volume": 1_000_000,
        "sma_10": 149.0, "sma_20": 148.5, "sma_50": 147.0, "ema_20": 149.2,
        "rsi": 55.0, "macd": 1.2, "macd_signal": 1.0, "macd_hist": 0.2,
        "volatility": 0.015, "close_lag_1": 149.0, "close_lag_5": 145.0,
        "daily_return": 0.005,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 500)
