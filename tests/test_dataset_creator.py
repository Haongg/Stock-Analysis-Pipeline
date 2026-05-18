import numpy as np
import pandas as pd

from dags.tasks.dataset_creator import LSTMDatasetCreator, create_lstm_sequences


def _make_df(n_rows: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n_rows, freq="D", tz="UTC")
    base = np.linspace(100, 140, n_rows)
    df = pd.DataFrame(
        {
            "@timestamp": ts,
            "ticker": ["AAPL"] * n_rows,
            "open": base,
            "high": base + 1.5,
            "low": base - 1.5,
            "close": base + 0.8,
            "volume": np.linspace(1_000, 2_000, n_rows),
            "sma_10": base,
            "sma_20": base,
            "sma_50": base,
            "ema_20": base,
            "rsi": np.linspace(30, 70, n_rows),
            "macd": np.linspace(-1, 1, n_rows),
            "macd_signal": np.linspace(-0.8, 0.8, n_rows),
            "macd_hist": np.linspace(-0.2, 0.2, n_rows),
            "volatility": np.linspace(0.01, 0.03, n_rows),
            "daily_return": np.linspace(-0.01, 0.01, n_rows),
            "close_lag_1": base - 0.5,
            "close_lag_5": base - 2.5,
        }
    )
    return df


def test_build_sequences_shape() -> None:
    creator = LSTMDatasetCreator(seq_length=30)
    df = _make_df(80)
    clean_df, _ = creator.apply_quality_filters(df)
    assert clean_df is not None
    X, y, ts = creator.build_sequences(clean_df)
    assert X.shape[1] == 30
    assert X.shape[2] == len(creator.feature_cols)
    assert y.shape[0] == X.shape[0]
    assert ts.shape[0] == X.shape[0]


def test_target_log_return() -> None:
    df = _make_df(40)
    df["is_imputed"] = 0
    X, y, _ = create_lstm_sequences(df, seq_length=30)
    assert X.shape[0] > 0
    # First y corresponds to log(close[30]/close[29])
    expected = np.log(float(df["close"].iloc[30]) / float(df["close"].iloc[29]))
    assert abs(float(y[0]) - expected) < 1e-10


def test_nan_filter_ratio() -> None:
    creator = LSTMDatasetCreator(seq_length=30, nan_ratio_threshold=0.01)
    df = _make_df(80)
    # inject heavy NaN ratio
    df.loc[:20, "sma_10"] = np.nan
    clean_df, report = creator.apply_quality_filters(df)
    assert clean_df is None
    assert report["reason"] == "nan_ratio_exceeded"


def test_nan_filter_consecutive() -> None:
    creator = LSTMDatasetCreator(seq_length=30, max_consecutive_nan=3)
    df = _make_df(80)
    df.loc[10:15, "rsi"] = np.nan
    clean_df, report = creator.apply_quality_filters(df)
    assert clean_df is None
    assert report["reason"] == "consecutive_nan_exceeded"


def test_is_imputed_feature_created() -> None:
    creator = LSTMDatasetCreator(seq_length=30)
    df = _make_df(80)
    df.loc[20, "ema_20"] = np.nan
    clean_df, report = creator.apply_quality_filters(df)
    assert clean_df is not None
    assert "is_imputed" in clean_df.columns
    assert int(clean_df["is_imputed"].sum()) >= 1
    assert report["imputed_count"] >= 1


def test_split_no_leakage_and_scaler_train_only() -> None:
    creator = LSTMDatasetCreator(seq_length=30)
    df = _make_df(100)
    clean_df, _ = creator.apply_quality_filters(df)
    assert clean_df is not None
    X, y, ts = creator.build_sequences(clean_df)
    split = creator.split_holdout(X, y, ts)
    assert split["train_ts"][-1] < split["val_ts"][0]
    assert split["val_ts"][-1] < split["test_ts"][0]

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = creator.fit_transform_scaler(
        split["X_train"], split["X_val"], split["X_test"]
    )
    assert X_train_scaled.shape == split["X_train"].shape
    assert X_val_scaled.shape == split["X_val"].shape
    assert X_test_scaled.shape == split["X_test"].shape
    assert scaler.mean_.shape[0] == len(creator.feature_cols)
