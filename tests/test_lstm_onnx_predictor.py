import math

import numpy as np

from src.flink.lstm_onnx_predictor import LSTMONNXPredictor
from src.flink.sequence_buffer import FEATURE_SEQUENCE_COLUMNS


class _FakeInput:
    name = "lstm_input"


class _FakeSession:
    def __init__(self, output):
        self.output = output
        self.inputs = []

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, output_names, inputs):
        self.inputs.append(inputs)
        return self.output


class _FakeScaler:
    def __init__(self):
        self.seen_shape = None

    def transform(self, values):
        self.seen_shape = values.shape
        return values + 1.0


def _predictor(output=None, sequence_length: int = 30) -> tuple[LSTMONNXPredictor, _FakeSession, _FakeScaler]:
    session = _FakeSession(output if output is not None else [np.array([[0.25]], dtype=np.float32)])
    scaler = _FakeScaler()
    predictor = LSTMONNXPredictor(
        sequence_length=sequence_length,
        feature_columns=FEATURE_SEQUENCE_COLUMNS,
        session=session,
        scaler=scaler,
    )
    return predictor, session, scaler


def _sequence(sequence_length: int = 30) -> np.ndarray:
    values = np.arange(sequence_length * len(FEATURE_SEQUENCE_COLUMNS), dtype=np.float32)
    return values.reshape(sequence_length, len(FEATURE_SEQUENCE_COLUMNS))


def _feature_record(sequence_length: int = 30) -> dict:
    matrix = _sequence(sequence_length)
    rows = []
    for row in matrix:
        rows.append({column: float(row[idx]) for idx, column in enumerate(FEATURE_SEQUENCE_COLUMNS)})
    return {
        "feature_sequence": rows,
        "feature_columns": list(FEATURE_SEQUENCE_COLUMNS),
    }


def test_predict_sequence_scales_2d_input_and_returns_float() -> None:
    predictor, session, scaler = _predictor()
    result = predictor.predict_sequence(_sequence())

    assert result == 0.25
    assert scaler.seen_shape == (30, len(FEATURE_SEQUENCE_COLUMNS))
    model_input = session.inputs[0]["lstm_input"]
    assert model_input.shape == (1, 30, len(FEATURE_SEQUENCE_COLUMNS))
    assert model_input.dtype == np.float32
    assert model_input[0, 0, 0] == 1.0


def test_predict_sequence_accepts_3d_input() -> None:
    predictor, session, _ = _predictor(output=[np.array([1.5], dtype=np.float32)])
    result = predictor.predict_sequence(_sequence().reshape(1, 30, len(FEATURE_SEQUENCE_COLUMNS)))

    assert result == 1.5
    assert session.inputs[0]["lstm_input"].shape == (1, 30, len(FEATURE_SEQUENCE_COLUMNS))


def test_predict_record_uses_configured_feature_order() -> None:
    predictor, _, _ = _predictor(output=[np.array([[2.75]], dtype=np.float32)])

    assert predictor.predict_record(_feature_record()) == 2.75


def test_invalid_shape_raises_value_error() -> None:
    predictor, _, _ = _predictor()
    bad_shape = np.zeros((29, len(FEATURE_SEQUENCE_COLUMNS)), dtype=np.float32)

    try:
        predictor.validate_sequence(bad_shape)
    except ValueError as exc:
        assert "Invalid sequence shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid sequence shape")


def test_nan_or_inf_raises_value_error() -> None:
    predictor, _, _ = _predictor()
    values = _sequence()
    values[0, 0] = math.nan

    try:
        predictor.validate_sequence(values)
    except ValueError as exc:
        assert "NaN or infinite" in str(exc)
    else:
        raise AssertionError("Expected ValueError for NaN input")


def test_predict_record_missing_feature_column_raises_value_error() -> None:
    predictor, _, _ = _predictor()
    record = _feature_record()
    del record["feature_sequence"][0][FEATURE_SEQUENCE_COLUMNS[0]]

    try:
        predictor.predict_record(record)
    except ValueError as exc:
        assert "missing required column" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing feature column")


def test_scalar_and_array_outputs_normalize_to_float() -> None:
    assert LSTMONNXPredictor._prediction_to_float([np.array([[3.5]])]) == 3.5
    assert LSTMONNXPredictor._prediction_to_float([np.array([4.5])]) == 4.5
    assert LSTMONNXPredictor._prediction_to_float([5.5]) == 5.5
