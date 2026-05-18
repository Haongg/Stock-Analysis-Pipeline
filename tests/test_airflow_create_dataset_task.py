import pandas as pd

from dags.tasks.dataset_creator import create_dataset_task_from_parquet_paths


def _make_valid_parquet(path: str, ticker: str = "AAPL", rows: int = 90) -> None:
    ts = pd.date_range("2025-01-01", periods=rows, freq="D", tz="UTC")
    base = pd.Series(range(rows), dtype="float64") + 100.0
    df = pd.DataFrame(
        {
            "@timestamp": ts,
            "ticker": [ticker] * rows,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": base * 10,
            "sma_10": base,
            "sma_20": base,
            "sma_50": base,
            "ema_20": base,
            "rsi": 50.0,
            "macd": 0.1,
            "macd_signal": 0.05,
            "macd_hist": 0.05,
            "volatility": 0.02,
            "daily_return": 0.001,
            "close_lag_1": base - 0.5,
            "close_lag_5": base - 2.5,
        }
    )
    df.to_parquet(path, index=False)


def test_create_dataset_task_returns_artifacts_for_valid_input(tmp_path) -> None:
    parquet_path = tmp_path / "features_AAPL_20260101.parquet"
    _make_valid_parquet(str(parquet_path), ticker="AAPL")
    artifacts = create_dataset_task_from_parquet_paths(
        parquet_paths=[str(parquet_path)],
        seq_length=30,
        output_dir=str(tmp_path),
    )
    assert len(artifacts) == 1
    assert artifacts[0].endswith(".pkl")


def test_create_dataset_task_raises_when_all_invalid(tmp_path) -> None:
    parquet_path = tmp_path / "features_BAD_20260101.parquet"
    _make_valid_parquet(str(parquet_path), ticker="BAD", rows=20)  # too short, should fail quality
    try:
        create_dataset_task_from_parquet_paths(
            parquet_paths=[str(parquet_path)],
            seq_length=30,
            output_dir=str(tmp_path),
        )
    except ValueError as exc:
        assert "No valid dataset artifacts" in str(exc)
        return
    raise AssertionError("Expected ValueError for invalid dataset inputs")
