"""Tests for the FastAPI inference endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Create a TestClient with the model pre-loaded via mock."""
    with patch("src.api.main.PredictionService.load_model"):
        from src.api.main import app, _service

        _service._model = MagicMock()
        _service._model.predict = MagicMock(return_value=[150.25])
        _service._model_version = "latest"

        yield TestClient(app)


def test_health_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_health_not_loaded():
    with patch("src.api.main.PredictionService.load_model"):
        from src.api.main import app, _service

        _service._model = None
        test_client = TestClient(app)
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "model_not_loaded"


def test_post_predict_valid_payload(client):
    payload = {
        "open": 150.0,
        "high": 155.0,
        "low": 148.0,
        "volume": 1_000_000,
        "sma_10": 149.0,
        "sma_20": 148.5,
        "sma_50": 147.0,
        "ema_20": 149.2,
        "rsi": 55.0,
        "macd": 1.2,
        "macd_signal": 1.0,
        "macd_hist": 0.2,
        "volatility": 0.015,
        "close_lag_1": 149.0,
        "close_lag_5": 145.0,
        "daily_return": 0.005,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_close" in data


def test_post_predict_missing_field(client):
    """Pydantic should reject payloads with missing required fields."""
    response = client.post("/predict", json={"open": 150.0})
    assert response.status_code == 422


def test_get_predict_ticker_model_not_ready():
    with patch("src.api.main.PredictionService.load_model"):
        from src.api.main import app, _service

        _service._model = None
        test_client = TestClient(app)
        response = test_client.get("/predict/AAPL")
        assert response.status_code == 503
