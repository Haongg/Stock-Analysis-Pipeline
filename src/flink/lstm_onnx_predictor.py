from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.flink.sequence_buffer import FEATURE_SEQUENCE_COLUMNS


class LSTMONNXPredictor:
    """Small ONNX inference wrapper for LSTM feature sequences."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        sequence_length: int = 30,
        feature_columns: Sequence[str] | None = None,
        session: Any | None = None,
        scaler: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.scaler_path = Path(scaler_path) if scaler_path is not None else None
        self.sequence_length = int(sequence_length)
        self.feature_columns = list(feature_columns or FEATURE_SEQUENCE_COLUMNS)
        self.session = session or self._load_session()
        self.scaler = scaler or self._load_scaler()
        self.input_name = self._resolve_input_name()

    def _load_session(self) -> Any:
        if self.model_path is None:
            raise ValueError("model_path is required when session is not provided.")
        import onnxruntime as ort

        return ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])

    def _load_scaler(self) -> Any:
        if self.scaler_path is None:
            raise ValueError("scaler_path is required when scaler is not provided.")
        with self.scaler_path.open("rb") as scaler_file:
            return pickle.load(scaler_file)

    def _resolve_input_name(self) -> str:
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError("ONNX session has no inputs.")
        return inputs[0].name

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    def validate_sequence(self, features_seq: Any) -> bool:
        array = np.asarray(features_seq)
        if array.ndim == 2:
            expected_shape = (self.sequence_length, self.n_features)
        elif array.ndim == 3:
            expected_shape = (1, self.sequence_length, self.n_features)
        else:
            raise ValueError(
                "features_seq must have shape "
                f"({self.sequence_length}, {self.n_features}) or "
                f"(1, {self.sequence_length}, {self.n_features})."
            )

        if tuple(array.shape) != expected_shape:
            raise ValueError(f"Invalid sequence shape {array.shape}; expected {expected_shape}.")
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("features_seq must contain numeric values.")
        if not np.isfinite(array.astype(np.float64)).all():
            raise ValueError("features_seq must not contain NaN or infinite values.")
        return True

    def prepare_input(self, features_seq: Any) -> np.ndarray:
        self.validate_sequence(features_seq)
        array = np.asarray(features_seq, dtype=np.float32)
        if array.ndim == 2:
            array = array.reshape(1, self.sequence_length, self.n_features)

        array_2d = array.reshape(-1, self.n_features)
        scaled_2d = self.scaler.transform(array_2d)
        return np.asarray(scaled_2d, dtype=np.float32).reshape(1, self.sequence_length, self.n_features)

    def predict_sequence(self, features_seq: Any) -> float:
        model_input = self.prepare_input(features_seq)
        outputs = self.session.run(None, {self.input_name: model_input})
        return self._prediction_to_float(outputs)

    def predict_record(self, record: dict) -> float:
        sequence = record.get("feature_sequence")
        if not sequence:
            raise ValueError("record must include non-empty feature_sequence.")

        feature_columns = record.get("feature_columns") or self.feature_columns
        missing_columns = [column for column in self.feature_columns if column not in feature_columns]
        if missing_columns:
            raise ValueError(f"record feature_columns missing required columns: {missing_columns}")

        matrix = []
        for row in sequence:
            try:
                matrix.append([row[column] for column in self.feature_columns])
            except KeyError as exc:
                raise ValueError(f"feature_sequence row missing required column: {exc.args[0]}") from exc
        return self.predict_sequence(matrix)

    @staticmethod
    def _prediction_to_float(outputs: Any) -> float:
        value = outputs
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("ONNX session returned no outputs.")
            value = value[0]

        array = np.asarray(value)
        if array.size == 0:
            raise ValueError("ONNX session returned an empty prediction output.")
        return float(array.reshape(-1)[0])
