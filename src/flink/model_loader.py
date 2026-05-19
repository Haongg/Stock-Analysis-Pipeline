from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from src.flink.error_handling import keep_previous_model
from src.flink.sequence_buffer import FEATURE_SEQUENCE_COLUMNS


try:
    from pyflink.datastream.functions import RichMapFunction
except Exception:
    class RichMapFunction:  # type: ignore[no-redef]
        """Fallback base class so unit tests do not require PyFlink."""

        pass


class ModelLoadError(RuntimeError):
    """Raised when a model artifact cannot be resolved or loaded."""


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelArtifactConfig:
    model_path: str = "models/lstm_latest.onnx"
    scaler_path: str = "models/feature_scaler.pkl"
    model_version: str = "local"
    sequence_length: int = 30
    feature_columns: Sequence[str] | None = None
    mlflow_model_uri: str | None = None
    mlflow_onnx_artifact_name: str = "model.onnx"
    mlflow_scaler_artifact_name: str = "feature_scaler.pkl"


@dataclass(frozen=True)
class LoadedModel:
    predictor: Any
    version: str
    model_path: str
    scaler_path: str
    loaded_at: datetime


@dataclass(frozen=True)
class ModelArtifactSignature:
    model_version: str
    model_path: str
    scaler_path: str
    model_mtime_ns: int | None
    scaler_mtime_ns: int | None
    mlflow_model_uri: str | None = None


@dataclass(frozen=True)
class ModelReloadSettings:
    enabled: bool = True
    interval_seconds: int = 300


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean.")


def resolve_model_artifact_config_from_env() -> ModelArtifactConfig:
    return ModelArtifactConfig(
        model_path=os.getenv("FLINK_MODEL_PATH", "models/lstm_latest.onnx"),
        scaler_path=os.getenv("FLINK_SCALER_PATH", "models/feature_scaler.pkl"),
        model_version=os.getenv("FLINK_MODEL_VERSION", "local"),
        sequence_length=_env_int("LSTM_SEQUENCE_LENGTH", 30),
        feature_columns=list(FEATURE_SEQUENCE_COLUMNS),
        mlflow_model_uri=os.getenv("MLFLOW_MODEL_URI") or None,
        mlflow_onnx_artifact_name=os.getenv("MLFLOW_ONNX_ARTIFACT_NAME", "model.onnx"),
        mlflow_scaler_artifact_name=os.getenv("MLFLOW_SCALER_ARTIFACT_NAME", "feature_scaler.pkl"),
    )


def resolve_model_reload_settings_from_env() -> ModelReloadSettings:
    interval_seconds = _env_int("FLINK_MODEL_RELOAD_INTERVAL_SECONDS", 300)
    if interval_seconds <= 0:
        raise ValueError("FLINK_MODEL_RELOAD_INTERVAL_SECONDS must be > 0.")
    return ModelReloadSettings(
        enabled=_env_bool("FLINK_MODEL_RELOAD_ENABLED", True),
        interval_seconds=interval_seconds,
    )


def _default_mlflow_download(uri: str) -> str:
    from mlflow.artifacts import download_artifacts

    return download_artifacts(artifact_uri=uri)


def _default_predictor_factory(**kwargs: Any) -> Any:
    from src.flink.lstm_onnx_predictor import LSTMONNXPredictor

    return LSTMONNXPredictor(**kwargs)


def resolve_model_artifact_paths(
    config: ModelArtifactConfig,
    download_fn: Callable[[str], str] | None = None,
) -> tuple[Path, Path]:
    if not config.mlflow_model_uri:
        return Path(config.model_path), Path(config.scaler_path)

    downloader = download_fn or _default_mlflow_download
    artifact_dir = Path(downloader(config.mlflow_model_uri))
    return artifact_dir / config.mlflow_onnx_artifact_name, artifact_dir / config.mlflow_scaler_artifact_name


def _path_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def build_model_artifact_signature(
    config: ModelArtifactConfig,
    download_fn: Callable[[str], str] | None = None,
) -> ModelArtifactSignature:
    model_path, scaler_path = resolve_model_artifact_paths(config, download_fn=download_fn)
    return ModelArtifactSignature(
        model_version=config.model_version,
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        model_mtime_ns=_path_mtime_ns(model_path),
        scaler_mtime_ns=_path_mtime_ns(scaler_path),
        mlflow_model_uri=config.mlflow_model_uri,
    )


def load_lstm_model(
    config: ModelArtifactConfig,
    predictor_factory: Callable[..., Any] = _default_predictor_factory,
    download_fn: Callable[[str], str] | None = None,
    loaded_at_fn: Callable[[], datetime] | None = None,
) -> LoadedModel:
    model_path, scaler_path = resolve_model_artifact_paths(config, download_fn=download_fn)
    missing_paths = [str(path) for path in (model_path, scaler_path) if not path.exists()]
    if missing_paths:
        raise ModelLoadError(f"Missing model artifact path(s): {', '.join(missing_paths)}")

    try:
        predictor = predictor_factory(
            model_path=model_path,
            scaler_path=scaler_path,
            sequence_length=config.sequence_length,
            feature_columns=config.feature_columns or FEATURE_SEQUENCE_COLUMNS,
        )
    except Exception as exc:
        raise ModelLoadError(f"Failed to load LSTM ONNX model: {exc}") from exc

    loaded_at = loaded_at_fn() if loaded_at_fn else datetime.now(timezone.utc)
    return LoadedModel(
        predictor=predictor,
        version=config.model_version,
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        loaded_at=loaded_at,
    )


def replace_loaded_model_or_keep_current(
    current_model: LoadedModel | None,
    config: ModelArtifactConfig,
    predictor_factory: Callable[..., Any] = _default_predictor_factory,
    download_fn: Callable[[str], str] | None = None,
) -> LoadedModel | None:
    try:
        candidate = load_lstm_model(config, predictor_factory=predictor_factory, download_fn=download_fn)
    except ModelLoadError:
        return keep_previous_model(current_model, None)
    return keep_previous_model(current_model, candidate)


class ModelReloadManager:
    """Polling-based hot reload manager for LSTM model artifacts."""

    def __init__(
        self,
        config: ModelArtifactConfig | None = None,
        config_resolver: Callable[[], ModelArtifactConfig] = resolve_model_artifact_config_from_env,
        reload_settings: ModelReloadSettings | None = None,
        reload_settings_resolver: Callable[[], ModelReloadSettings] = resolve_model_reload_settings_from_env,
        predictor_factory: Callable[..., Any] = _default_predictor_factory,
        download_fn: Callable[[str], str] | None = None,
        loaded_at_fn: Callable[[], datetime] | None = None,
        clock_fn: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.config_resolver = config_resolver
        self.reload_settings = reload_settings
        self.reload_settings_resolver = reload_settings_resolver
        self.predictor_factory = predictor_factory
        self.download_fn = download_fn
        self.loaded_at_fn = loaded_at_fn
        self.clock_fn = clock_fn
        self.logger = logger or LOGGER
        self.current_model: LoadedModel | None = None
        self.current_signature: ModelArtifactSignature | None = None
        self.last_reload_check: float | None = None
        self.last_error: str | None = None

    def _resolve_config(self) -> ModelArtifactConfig:
        return self.config or self.config_resolver()

    def _resolve_settings(self) -> ModelReloadSettings:
        return self.reload_settings or self.reload_settings_resolver()

    def load_initial(self) -> LoadedModel:
        config = self._resolve_config()
        signature = build_model_artifact_signature(config, download_fn=self.download_fn)
        loaded_model = load_lstm_model(
            config,
            predictor_factory=self.predictor_factory,
            download_fn=self.download_fn,
            loaded_at_fn=self.loaded_at_fn,
        )
        self.current_model = loaded_model
        self.current_signature = signature
        self.last_reload_check = self.clock_fn()
        self.last_error = None
        self.logger.info("Loaded LSTM model version %s from %s", loaded_model.version, loaded_model.model_path)
        return loaded_model

    def should_check_reload(self, now: float | None = None) -> bool:
        settings = self._resolve_settings()
        if not settings.enabled:
            return False
        current_time = self.clock_fn() if now is None else now
        if self.last_reload_check is None:
            return True
        return current_time - self.last_reload_check >= settings.interval_seconds

    def maybe_reload(self) -> LoadedModel | None:
        settings = self._resolve_settings()
        if not settings.enabled:
            return self.current_model

        now = self.clock_fn()
        if not self.should_check_reload(now):
            return self.current_model

        self.last_reload_check = now
        config = self._resolve_config()
        try:
            candidate_signature = build_model_artifact_signature(config, download_fn=self.download_fn)
        except Exception as exc:
            self.last_error = str(exc)
            self.logger.warning("Unable to resolve candidate LSTM model artifacts: %s", exc)
            return self.current_model

        if self.current_model is not None and candidate_signature == self.current_signature:
            return self.current_model

        self.logger.info("Attempting LSTM model reload for version %s", config.model_version)
        try:
            candidate = load_lstm_model(
                config,
                predictor_factory=self.predictor_factory,
                download_fn=self.download_fn,
                loaded_at_fn=self.loaded_at_fn,
            )
        except ModelLoadError as exc:
            self.last_error = str(exc)
            self.current_model = keep_previous_model(self.current_model, None)
            self.logger.warning("LSTM model reload failed; preserving previous model: %s", exc)
            return self.current_model

        previous_version = self.current_model.version if self.current_model is not None else None
        self.current_model = keep_previous_model(self.current_model, candidate)
        self.current_signature = candidate_signature
        self.last_error = None
        self.logger.info("Swapped LSTM model version from %s to %s", previous_version, candidate.version)
        return self.current_model


class ModelLoaderRichMapFunction(RichMapFunction):
    """Startup model loader scaffold for future Flink inference integration."""

    def __init__(
        self,
        config: ModelArtifactConfig | None = None,
        predictor_factory: Callable[..., Any] = _default_predictor_factory,
        download_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config
        self.predictor_factory = predictor_factory
        self.download_fn = download_fn
        self.loaded_model: LoadedModel | None = None

    def open(self, runtime_context: Any = None) -> None:
        config = self.config or resolve_model_artifact_config_from_env()
        self.loaded_model = load_lstm_model(
            config,
            predictor_factory=self.predictor_factory,
            download_fn=self.download_fn,
        )

    def get_loaded_model(self) -> LoadedModel | None:
        return self.loaded_model

    def get_predictor(self) -> Any | None:
        return self.loaded_model.predictor if self.loaded_model is not None else None

    def map(self, record: dict) -> dict:
        output = dict(record)
        if self.loaded_model is not None:
            output.setdefault("model_version", self.loaded_model.version)
        return output
