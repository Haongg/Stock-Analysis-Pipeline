"""Tests for FeatureEngineer interface."""

import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer


def test_transform_raises_not_implemented():
    engineer = FeatureEngineer()
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(NotImplementedError):
        engineer.transform(df)
