from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


BASE_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "volatility",
    "daily_return",
    "close_lag_1",
    "close_lag_5",
]


def create_lstm_sequences(
    df: pd.DataFrame, seq_length: int = 30, feature_cols: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feature_cols is None:
        feature_cols = BASE_FEATURE_COLUMNS + ["is_imputed"]

    X: list[np.ndarray] = []
    y: list[float] = []
    ts: list[np.datetime64] = []

    values = df[feature_cols].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    timestamps = df["@timestamp"].to_numpy()

    # Use next-day log-return as target.
    # X window ends at i-1, target is log(close[i] / close[i-1]).
    for i in range(seq_length, len(df)):
        if i - 1 < 0:
            continue
        prev_close = close[i - 1]
        curr_close = close[i]
        if prev_close <= 0 or curr_close <= 0:
            continue
        X.append(values[i - seq_length : i])
        y.append(float(np.log(curr_close / prev_close)))
        ts.append(timestamps[i])

    if not X:
        return np.empty((0, seq_length, len(feature_cols))), np.empty((0,)), np.empty((0,))

    return np.stack(X), np.asarray(y, dtype=float), np.asarray(ts)


class LSTMDatasetCreator:
    def __init__(
        self,
        seq_length: int = 30,
        nan_ratio_threshold: float = 0.01,
        max_consecutive_nan: int = 3,
        fill_limit: int = 2,
        output_dir: str = "data",
    ) -> None:
        self.seq_length = seq_length
        self.nan_ratio_threshold = nan_ratio_threshold
        self.max_consecutive_nan = max_consecutive_nan
        self.fill_limit = fill_limit
        self.output_dir = output_dir
        self.feature_cols = BASE_FEATURE_COLUMNS + ["is_imputed"]
        self.target_col = "log_return_next_day"

    def load_parquet(self, path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if "@timestamp" not in df.columns:
            raise ValueError(f"Missing @timestamp in parquet: {path}")
        df["@timestamp"] = pd.to_datetime(df["@timestamp"], utc=True, errors="coerce")
        df = df.sort_values("@timestamp").drop_duplicates(subset=["@timestamp"], keep="last")
        df = df.reset_index(drop=True)
        return df

    def _max_consecutive_true(self, mask: pd.Series) -> int:
        max_run = 0
        run = 0
        for val in mask.astype(bool).tolist():
            if val:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        return max_run

    def apply_quality_filters(self, df: pd.DataFrame) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        required_cols = ["@timestamp", "ticker", "close"] + BASE_FEATURE_COLUMNS
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return None, {"reason": "missing_columns", "missing_columns": missing_cols}

        work_df = df.copy()
        feature_df = work_df[BASE_FEATURE_COLUMNS]
        nan_ratio_before = float(feature_df.isna().mean().mean())

        if nan_ratio_before > self.nan_ratio_threshold:
            return None, {"reason": "nan_ratio_exceeded", "nan_ratio_before": nan_ratio_before}

        consecutive_nan_max = 0
        for col in BASE_FEATURE_COLUMNS:
            consecutive_nan_max = max(consecutive_nan_max, self._max_consecutive_true(work_df[col].isna()))
        if consecutive_nan_max > self.max_consecutive_nan:
            return None, {
                "reason": "consecutive_nan_exceeded",
                "consecutive_nan_max": consecutive_nan_max,
            }

        # Flag rows that had any missing feature before imputation.
        imputed_mask = work_df[BASE_FEATURE_COLUMNS].isna().any(axis=1)
        work_df["is_imputed"] = imputed_mask.astype(int)

        # Forward-fill with strict limit.
        work_df[BASE_FEATURE_COLUMNS] = work_df[BASE_FEATURE_COLUMNS].ffill(limit=self.fill_limit)

        # Drop rows where core features still null after fill.
        core_cols = ["open", "high", "low", "close", "volume"]
        rows_before_drop = len(work_df)
        work_df = work_df.dropna(subset=core_cols)
        rows_dropped = rows_before_drop - len(work_df)

        if work_df.empty:
            return None, {"reason": "empty_after_drop"}

        if work_df[BASE_FEATURE_COLUMNS].isna().any().any():
            return None, {"reason": "nan_remaining_after_fill"}

        if len(work_df) < self.seq_length + 2:
            return None, {"reason": "insufficient_rows_after_clean", "rows": len(work_df)}

        nan_ratio_after = float(work_df[BASE_FEATURE_COLUMNS].isna().mean().mean())
        quality_report = {
            "nan_ratio_before": nan_ratio_before,
            "nan_ratio_after": nan_ratio_after,
            "rows_before": len(df),
            "rows_after": len(work_df),
            "rows_dropped": rows_dropped,
            "consecutive_nan_max": consecutive_nan_max,
            "imputed_count": int(work_df["is_imputed"].sum()),
        }
        return work_df.reset_index(drop=True), quality_report

    def build_sequences(self, df: pd.DataFrame, seq_length: int | None = None):
        return create_lstm_sequences(df, seq_length=seq_length or self.seq_length, feature_cols=self.feature_cols)

    def split_holdout(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ts: np.ndarray,
        ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> dict[str, np.ndarray]:
        if X.shape[0] == 0:
            raise ValueError("No samples available for split.")
        train_ratio, val_ratio, test_ratio = ratios
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
            raise ValueError("Split ratios must sum to 1.0")

        n = X.shape[0]
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        if train_end <= 0 or val_end <= train_end or val_end >= n:
            raise ValueError("Invalid split boundaries for current sample size.")

        return {
            "X_train": X[:train_end],
            "y_train": y[:train_end],
            "train_ts": ts[:train_end],
            "X_val": X[train_end:val_end],
            "y_val": y[train_end:val_end],
            "val_ts": ts[train_end:val_end],
            "X_test": X[val_end:],
            "y_test": y[val_end:],
            "test_ts": ts[val_end:],
        }

    def fit_transform_scaler(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        feature_cols: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        feature_cols = feature_cols or self.feature_cols
        scaler = StandardScaler()
        n_features = len(feature_cols)

        train_2d = X_train.reshape(-1, n_features)
        val_2d = X_val.reshape(-1, n_features)
        test_2d = X_test.reshape(-1, n_features)

        scaler.fit(train_2d)
        X_train_scaled = scaler.transform(train_2d).reshape(X_train.shape)
        X_val_scaled = scaler.transform(val_2d).reshape(X_val.shape)
        X_test_scaled = scaler.transform(test_2d).reshape(X_test.shape)
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler

    def save_artifact(
        self,
        ticker: str,
        split_data: dict[str, np.ndarray],
        scaler: StandardScaler,
        quality_report: dict[str, Any],
        run_date: str | None = None,
    ) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        date_part = run_date or datetime.now(timezone.utc).strftime("%Y%m%d")
        output_path = os.path.join(self.output_dir, f"lstm_dataset_{ticker.upper()}_{date_part}.pkl")
        artifact = {
            "ticker": ticker.upper(),
            "seq_length": self.seq_length,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "X_train": split_data["X_train"],
            "y_train": split_data["y_train"],
            "X_val": split_data["X_val"],
            "y_val": split_data["y_val"],
            "X_test": split_data["X_test"],
            "y_test": split_data["y_test"],
            "train_ts": split_data["train_ts"],
            "val_ts": split_data["val_ts"],
            "test_ts": split_data["test_ts"],
            "scaler": scaler,
            "quality_report": quality_report,
            "n_samples": int(
                split_data["X_train"].shape[0]
                + split_data["X_val"].shape[0]
                + split_data["X_test"].shape[0]
            ),
        }
        with open(output_path, "wb") as f:
            pickle.dump(artifact, f)
        return output_path

    def build_and_save_dataset(self, parquet_path: str) -> str | None:
        df = self.load_parquet(parquet_path)
        if df.empty:
            logger.warning("Skipping empty parquet path=%s", parquet_path)
            return None

        ticker = str(df["ticker"].iloc[0]).upper()
        clean_df, quality_report = self.apply_quality_filters(df)
        if clean_df is None:
            logger.warning("Skipping ticker=%s due to quality filter: %s", ticker, quality_report)
            return None

        X, y, ts = self.build_sequences(clean_df)
        if X.shape[0] == 0:
            logger.warning("Skipping ticker=%s due to empty sequences", ticker)
            return None

        split_data = self.split_holdout(X, y, ts)
        X_train, X_val, X_test, scaler = self.fit_transform_scaler(
            split_data["X_train"], split_data["X_val"], split_data["X_test"]
        )
        split_data["X_train"] = X_train
        split_data["X_val"] = X_val
        split_data["X_test"] = X_test

        return self.save_artifact(ticker=ticker, split_data=split_data, scaler=scaler, quality_report=quality_report)


def create_dataset_task_from_parquet_paths(
    parquet_paths: list[str], seq_length: int = 30, output_dir: str = "data"
) -> list[str]:
    creator = LSTMDatasetCreator(seq_length=seq_length, output_dir=output_dir)
    artifact_paths: list[str] = []
    for parquet_path in parquet_paths:
        artifact_path = creator.build_and_save_dataset(parquet_path)
        if artifact_path is not None:
            artifact_paths.append(artifact_path)

    if not artifact_paths:
        raise ValueError("No valid dataset artifacts created from provided parquet paths.")

    return artifact_paths
