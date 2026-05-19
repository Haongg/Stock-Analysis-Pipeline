"""
models/trainer.py

Responsibility: train the stock-price prediction model and log everything
to MLflow (parameters, metrics, model artifact, model alias).

Model: LSTM (Long Short-Term Memory) for time-series stock price prediction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

# Feature columns expected in the input DataFrame
FEATURE_COLS = [
    "SMA10", "SMA20", "SMA50", "EMA20", "RSI_14", "MACD",
    "Signal_Line", "Histogram", "Daily_Return", "Rolling_Volatility",
    "Close_lag_1", "Close_lag_5"
]
TARGET_COL = "Close"

DEFAULT_PARAMS: Dict[str, Any] = {
    "sequence_length": 60,
    "lstm_units": [128, 64],
    "dropout_rate": 0.2,
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "random_state": 42,
}


class StockModelTrainer:
    """Trains and registers an LSTM model for stock price prediction."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: str = "stock_analysis",
    ) -> None:
        self.params = params or DEFAULT_PARAMS.copy()
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def train(
        self, df: pd.DataFrame
    ) -> Tuple[keras.Model, Dict[str, float]]:
        
        X_train, y_train, X_test, y_test, X_validation, y_validation = self._prepare_data(df)
        
        # Build LSTM model
        self.model = self._build_lstm_model(X_train.shape[1], X_train.shape[2])
        
        # Train the model
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_validation, y_validation),
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            callbacks=[early_stop],
            verbose=1
        )

        # Compute metrics
        metrics_val = self._compute_metrics(y_validation, self.model.predict(X_validation))
        metrics_test = self._compute_metrics(y_test, self.model.predict(X_test))
        
        return self.model, {**metrics_val, **metrics_test}

    def _build_lstm_model(self, time_steps: int, n_features: int) -> keras.Model:
        """Build LSTM neural network model."""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.params["lstm_units"][0],
            activation='relu',
            return_sequences=True,
            input_shape=(time_steps, n_features)
        ))
        model.add(Dropout(self.params["dropout_rate"]))
        
        # Second LSTM layer
        model.add(LSTM(
            units=self.params["lstm_units"][1],
            activation='relu',
            return_sequences=False
        ))
        model.add(Dropout(self.params["dropout_rate"]))
        
        # Dense layers
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=1))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.params["learning_rate"])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model

    def _prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for LSTM model with sequences."""
        # Extract features and target
        features = df[FEATURE_COLS].copy()
        target = df[[TARGET_COL]].copy()
        
        # Handle missing values
        valid_mask = features.notna().all(axis=1) & target.notna().all(axis=1)
        features = features[valid_mask].reset_index(drop=True)
        target = target[valid_mask].reset_index(drop=True)
        
        # Normalize features and target
        features_scaled = self.scaler.fit_transform(features)
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaled = target_scaler.fit_transform(target)
        
        # Create sequences
        seq_length = self.params["sequence_length"]
        X, y = [], []
        
        for i in range(len(features_scaled) - seq_length):
            X.append(features_scaled[i:i+seq_length])
            y.append(target_scaled[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data into train, validation and test sets
        train_ratio = 0.85
        valid_ratio = 0.05
        
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        valid_end = train_end + int(n_samples * valid_ratio)
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_validation = X[train_end:valid_end]
        y_validation = y[train_end:valid_end]
        X_test = X[valid_end:]
        y_test = y[valid_end:]
        
        return X_train, y_train, X_test, y_test, X_validation, y_validation

    @staticmethod
    def _compute_metrics(y_true, y_pred) -> Dict[str, float]:
        """Return dict with keys ``rmse``, ``mae``, ``r2``."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
