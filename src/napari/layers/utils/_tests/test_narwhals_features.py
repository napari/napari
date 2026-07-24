"""Tests for the new narwhals/polars feature paths not already covered
in test_layer_utils.py (which tests dict, None, and pandas paths)."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from napari.layers import Labels, Points, Shapes, Surface, Tracks, Vectors
from napari.layers.utils.layer_utils import (
    _features_from_properties,
    _FeatureTable,
    _to_pandas,
    _validate_feature_defaults,
    _validate_features,
    dataframe_to_properties,
    validate_properties,
)


class TestPolarsInput:
    """Functions that accept ``IntoDataFrame`` work with Polars input."""

    def test_validate_features(self):
        result = _validate_features(
            pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['a', 'b']

    def test_feature_table(self):
        ft = _FeatureTable(
            pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        )
        assert isinstance(ft.values, pd.DataFrame)

    def test_dataframe_to_properties(self):
        result = dataframe_to_properties(
            pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        )
        assert isinstance(result, dict)

    def test_validate_properties(self):
        result = validate_properties(
            pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        )
        assert isinstance(result, dict)


def test_validate_feature_defaults_polars():
    """Non-dict defaults trigger the _to_pandas conversion branch."""
    ft = _FeatureTable(pl.DataFrame({'x': [10, 20, 30], 'y': ['a', 'b', 'c']}))
    defaults = _validate_feature_defaults(
        pl.DataFrame({'x': [99], 'y': ['z']}), ft.values
    )
    assert isinstance(defaults, pd.DataFrame)


def test_features_from_properties_polars():
    """Non-dict properties with property_choices trigger the conversion."""
    result = _features_from_properties(
        properties=pl.DataFrame({'x': [10, 20, 30], 'y': ['a', 'b', 'c']}),
        property_choices={'x': np.array([10, 20, 30])},
    )
    assert isinstance(result, pd.DataFrame)


def test_to_pandas():
    result = _to_pandas(
        pl.DataFrame({'x': [10, 20, 30], 'y': ['a', 'b', 'c']})
    )
    assert isinstance(result, pd.DataFrame)


def test_to_pandas_without_pyarrow(monkeypatch):
    import narwhals as nw

    def raise_import_error(_):
        raise ImportError

    monkeypatch.setattr(nw.DataFrame, 'to_pandas', raise_import_error)
    result = _to_pandas(
        pl.DataFrame({'x': [10, 20, 30], 'y': ['a', 'b', 'c']})
    )
    pd.testing.assert_frame_equal(
        result, pd.DataFrame({'x': [10, 20, 30], 'y': ['a', 'b', 'c']})
    )


class TestLayerIntegration:
    """Each concrete layer accepts Polars features."""

    @pytest.mark.parametrize(('layer_cls', 'data', 'feat'), [
        pytest.param(Points, np.array([[0, 0], [1, 1]]), {'category': ['a', 'b']}, id='Points'),
        pytest.param(Labels, np.array([[0, 1], [1, 0]]), {'category': ['a', 'b']}, id='Labels'),
        pytest.param(
            Shapes,
            np.array([[[0, 0], [1, 0], [1, 1], [0, 1]], [[2, 2], [3, 2], [3, 3], [2, 3]]]),
            {'category': ['a', 'b']},
            id='Shapes',
        ),
        pytest.param(
            Surface,
            (np.array([[0, 0], [1, 0], [0, 1]]), np.array([[0, 1, 2]]), np.array([1, 2, 3])),
            {'category': ['a', 'b', 'c']},
            id='Surface',
        ),
        pytest.param(Tracks, np.array([[0, 0, 0, 0], [0, 1, 1, 1]]), {'category': ['a', 'b']}, id='Tracks'),
        pytest.param(Vectors, np.array([[[0, 0], [1, 1]], [[1, 1], [2, 2]]]), {'category': ['a', 'b']}, id='Vectors'),
    ])
    def test_features_polars(self, layer_cls, data, feat):
        features = pl.DataFrame(feat)
        layer = layer_cls(data, features=features)
        pd.testing.assert_frame_equal(layer.features, pd.DataFrame(feat))
