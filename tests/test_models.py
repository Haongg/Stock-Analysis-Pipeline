"""Tests for StockModelTrainer interface."""

import pandas as pd
import pytest

from src.models.trainer import FEATURE_COLS, StockModelTrainer


def test_train_raises_not_implemented():
    trainer = StockModelTrainer()
    df = pd.DataFrame({col: [1.0] * 10 for col in FEATURE_COLS + ["close"]})
    with pytest.raises(NotImplementedError):
        trainer.train(df)


def test_compute_metrics_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        StockModelTrainer._compute_metrics([1.0], [1.0])
