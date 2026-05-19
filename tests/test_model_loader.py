import os
from datetime import datetime, timezone

from src.flink.model_loader import (
    LoadedModel,
    ModelArtifactConfig,
    ModelReloadManager,
    ModelReloadSettings,
    ModelLoaderRichMapFunction,
    ModelLoadError,
    build_model_artifact_signature,
    load_lstm_model,
    replace_loaded_model_or_keep_current,
    resolve_model_artifact_config_from_env,
    resolve_model_reload_settings_from_env,
    resolve_model_artifact_paths,
)
from src.flink.sequence_buffer import FEATURE_SEQUENCE_COLUMNS


class _FakePredictor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _factory(**kwargs):
    return _FakePredictor(**kwargs)


class _Clock:
    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _artifact_config(tmp_path, version: str = "v1") -> ModelArtifactConfig:
    model_path = tmp_path / f"model_{version}.onnx"
    scaler_path = tmp_path / f"scaler_{version}.pkl"
    model_path.write_bytes(b"model")
    scaler_path.write_bytes(b"scaler")
    return ModelArtifactConfig(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        model_version=version,
        feature_columns=FEATURE_SEQUENCE_COLUMNS,
    )


def test_env_resolver_returns_defaults(monkeypatch) -> None:
    for key in [
        "FLINK_MODEL_PATH",
        "FLINK_SCALER_PATH",
        "FLINK_MODEL_VERSION",
        "LSTM_SEQUENCE_LENGTH",
        "MLFLOW_MODEL_URI",
    ]:
        monkeypatch.delenv(key, raising=False)

    config = resolve_model_artifact_config_from_env()

    assert config.model_path == "models/lstm_latest.onnx"
    assert config.scaler_path == "models/feature_scaler.pkl"
    assert config.model_version == "local"
    assert config.sequence_length == 30
    assert list(config.feature_columns) == FEATURE_SEQUENCE_COLUMNS


def test_env_resolver_respects_overrides(monkeypatch) -> None:
    monkeypatch.setenv("FLINK_MODEL_PATH", "models/custom.onnx")
    monkeypatch.setenv("FLINK_SCALER_PATH", "models/custom_scaler.pkl")
    monkeypatch.setenv("FLINK_MODEL_VERSION", "v9")
    monkeypatch.setenv("LSTM_SEQUENCE_LENGTH", "45")
    monkeypatch.setenv("MLFLOW_MODEL_URI", "models:/lstm/9")

    config = resolve_model_artifact_config_from_env()

    assert config.model_path == "models/custom.onnx"
    assert config.scaler_path == "models/custom_scaler.pkl"
    assert config.model_version == "v9"
    assert config.sequence_length == 45
    assert config.mlflow_model_uri == "models:/lstm/9"


def test_reload_settings_resolver_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.delenv("FLINK_MODEL_RELOAD_ENABLED", raising=False)
    monkeypatch.delenv("FLINK_MODEL_RELOAD_INTERVAL_SECONDS", raising=False)

    defaults = resolve_model_reload_settings_from_env()

    assert defaults.enabled is True
    assert defaults.interval_seconds == 300

    monkeypatch.setenv("FLINK_MODEL_RELOAD_ENABLED", "false")
    monkeypatch.setenv("FLINK_MODEL_RELOAD_INTERVAL_SECONDS", "15")

    overrides = resolve_model_reload_settings_from_env()

    assert overrides.enabled is False
    assert overrides.interval_seconds == 15


def test_local_artifacts_load_loaded_model(tmp_path) -> None:
    model_path = tmp_path / "model.onnx"
    scaler_path = tmp_path / "scaler.pkl"
    model_path.write_bytes(b"model")
    scaler_path.write_bytes(b"scaler")
    loaded_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    loaded = load_lstm_model(
        ModelArtifactConfig(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            model_version="v1",
            sequence_length=30,
            feature_columns=FEATURE_SEQUENCE_COLUMNS,
        ),
        predictor_factory=_factory,
        loaded_at_fn=lambda: loaded_at,
    )

    assert loaded.version == "v1"
    assert loaded.model_path == str(model_path)
    assert loaded.scaler_path == str(scaler_path)
    assert loaded.loaded_at == loaded_at
    assert loaded.predictor.kwargs["model_path"] == model_path
    assert loaded.predictor.kwargs["scaler_path"] == scaler_path


def test_missing_artifacts_raise_model_load_error(tmp_path) -> None:
    try:
        load_lstm_model(
            ModelArtifactConfig(
                model_path=str(tmp_path / "missing.onnx"),
                scaler_path=str(tmp_path / "missing.pkl"),
            ),
            predictor_factory=_factory,
        )
    except ModelLoadError as exc:
        assert "Missing model artifact" in str(exc)
    else:
        raise AssertionError("Expected ModelLoadError for missing artifacts")


def test_failed_candidate_load_preserves_previous_loaded_model(tmp_path) -> None:
    previous = LoadedModel(
        predictor=_FakePredictor(),
        version="old",
        model_path="old.onnx",
        scaler_path="old.pkl",
        loaded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    result = replace_loaded_model_or_keep_current(
        previous,
        ModelArtifactConfig(model_path=str(tmp_path / "missing.onnx"), scaler_path=str(tmp_path / "missing.pkl")),
        predictor_factory=_factory,
    )

    assert result is previous


def test_reload_does_not_happen_before_interval_elapses(tmp_path) -> None:
    clock = _Clock()
    configs = [_artifact_config(tmp_path, "v1")]
    manager = ModelReloadManager(
        config_resolver=lambda: configs[-1],
        reload_settings=ModelReloadSettings(enabled=True, interval_seconds=300),
        predictor_factory=_factory,
        clock_fn=clock,
    )

    first = manager.load_initial()
    configs.append(_artifact_config(tmp_path, "v2"))
    clock.advance(299)
    result = manager.maybe_reload()

    assert result is first
    assert manager.current_model is first


def test_changed_model_version_triggers_reload_after_interval(tmp_path) -> None:
    clock = _Clock()
    configs = [_artifact_config(tmp_path, "v1")]
    manager = ModelReloadManager(
        config_resolver=lambda: configs[-1],
        reload_settings=ModelReloadSettings(enabled=True, interval_seconds=300),
        predictor_factory=_factory,
        clock_fn=clock,
    )

    manager.load_initial()
    configs.append(_artifact_config(tmp_path, "v2"))
    clock.advance(300)
    result = manager.maybe_reload()

    assert result is not None
    assert result.version == "v2"
    assert manager.last_error is None


def test_changed_artifact_mtime_changes_signature(tmp_path) -> None:
    config = _artifact_config(tmp_path, "v1")
    first = build_model_artifact_signature(config)
    model_path = tmp_path / "model_v1.onnx"
    model_path.write_bytes(b"new model")
    assert first.model_mtime_ns is not None
    os.utime(model_path, ns=(first.model_mtime_ns + 1_000_000_000, first.model_mtime_ns + 1_000_000_000))
    second = build_model_artifact_signature(config)

    assert second.model_path == first.model_path
    assert second.model_mtime_ns != first.model_mtime_ns


def test_failed_reload_preserves_previous_loaded_model(tmp_path) -> None:
    clock = _Clock()
    configs = [_artifact_config(tmp_path, "v1")]
    manager = ModelReloadManager(
        config_resolver=lambda: configs[-1],
        reload_settings=ModelReloadSettings(enabled=True, interval_seconds=300),
        predictor_factory=_factory,
        clock_fn=clock,
    )

    previous = manager.load_initial()
    configs.append(
        ModelArtifactConfig(
            model_path=str(tmp_path / "missing.onnx"),
            scaler_path=str(tmp_path / "missing.pkl"),
            model_version="v2",
        )
    )
    clock.advance(300)
    result = manager.maybe_reload()

    assert result is previous
    assert manager.current_model is previous
    assert manager.last_error is not None
    assert "Missing model artifact" in manager.last_error


def test_disabled_reload_returns_current_model_unchanged(tmp_path) -> None:
    clock = _Clock()
    configs = [_artifact_config(tmp_path, "v1")]
    manager = ModelReloadManager(
        config_resolver=lambda: configs[-1],
        reload_settings=ModelReloadSettings(enabled=False, interval_seconds=1),
        predictor_factory=_factory,
        clock_fn=clock,
    )

    previous = manager.load_initial()
    configs.append(_artifact_config(tmp_path, "v2"))
    clock.advance(10)
    result = manager.maybe_reload()

    assert result is previous
    assert manager.current_model.version == "v1"


def test_rich_map_function_loads_and_attaches_model_version(tmp_path) -> None:
    model_path = tmp_path / "model.onnx"
    scaler_path = tmp_path / "scaler.pkl"
    model_path.write_bytes(b"model")
    scaler_path.write_bytes(b"scaler")
    loader = ModelLoaderRichMapFunction(
        config=ModelArtifactConfig(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            model_version="v2",
        ),
        predictor_factory=_factory,
    )

    loader.open()
    output = loader.map({"ticker": "AAPL"})

    assert loader.get_loaded_model() is not None
    assert loader.get_predictor() is loader.get_loaded_model().predictor
    assert output == {"ticker": "AAPL", "model_version": "v2"}


def test_mlflow_artifact_paths_use_injected_downloader(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    config = ModelArtifactConfig(
        mlflow_model_uri="models:/lstm/1",
        mlflow_onnx_artifact_name="candidate.onnx",
        mlflow_scaler_artifact_name="candidate_scaler.pkl",
    )

    model_path, scaler_path = resolve_model_artifact_paths(config, download_fn=lambda uri: str(artifact_dir))

    assert model_path == artifact_dir / "candidate.onnx"
    assert scaler_path == artifact_dir / "candidate_scaler.pkl"
