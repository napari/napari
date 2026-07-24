"""Tests verifying the narwhals ``IntoDataFrame`` contract.

Every feature/property boundary function that accepts ``IntoDataFrame`` is
tested with both pandas and polars to ensure compatibility stays in place.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from napari.layers import Labels, Points, Shapes, Surface, Tracks, Vectors
from napari.layers.utils.layer_utils import (
    _features_from_properties,
    _features_to_properties,
    _FeatureTable,
    _to_pandas,
    _validate_feature_defaults,
    _validate_features,
    dataframe_to_properties,
    validate_properties,
)

# ---------------------------------------------------------------------------
# Parametrized fixture — each test runs once per backend
# ---------------------------------------------------------------------------

_BACKENDS = [
    pytest.param(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), id='pandas'),
    pytest.param(pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), id='polars'),
]


@pytest.fixture(params=_BACKENDS)
def any_df(request):
    return request.param


class TestBackendAgnostic:
    """Boundary functions that must accept any eager DataFrame."""

    def test_validate_features(self, any_df):
        result = _validate_features(any_df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['a', 'b']

    def test_feature_table(self, any_df):
        ft = _FeatureTable(any_df)
        assert isinstance(ft.values, pd.DataFrame)

    def test_dataframe_to_properties(self, any_df):
        result = dataframe_to_properties(any_df)
        assert isinstance(result, dict)
        assert 'a' in result

    def test_validate_properties(self, any_df):
        result = validate_properties(any_df)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Non-backend inputs (dict, None)
# ---------------------------------------------------------------------------


def test_validate_features_dict():
    result = _validate_features({'a': [1, 2, 3]})
    assert isinstance(result, pd.DataFrame)


def test_validate_features_none():
    result = _validate_features(None)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_feature_table_dict():
    ft = _FeatureTable({'a': [1, 2, 3]})
    assert isinstance(ft.values, pd.DataFrame)


def test_features_to_properties():
    result = _features_to_properties(pd.DataFrame({'a': [1, 2, 3]}))
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Polars-specific — code paths that only activate with non-pandas input
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Layer integration — each concrete layer accepts Polars features
# ---------------------------------------------------------------------------

_LAYER_CASES = [
    pytest.param(
        Points,
        np.array([[0, 0], [1, 1]]),
        {'category': ['a', 'b']},
        id='Points',
    ),
    pytest.param(
        Labels,
        np.array([[0, 1], [1, 0]]),
        {'category': ['a', 'b']},
        id='Labels',
    ),
    pytest.param(
        Shapes,
        np.array(
            [
                [[0, 0], [1, 0], [1, 1], [0, 1]],
                [[2, 2], [3, 2], [3, 3], [2, 3]],
            ]
        ),
        {'category': ['a', 'b']},
        id='Shapes',
    ),
    pytest.param(
        Surface,
        (
            np.array([[0, 0], [1, 0], [0, 1]]),
            np.array([[0, 1, 2]]),
            np.array([1, 2, 3]),
        ),
        {'category': ['a', 'b', 'c']},
        id='Surface',
    ),
    pytest.param(
        Tracks,
        np.array([[0, 0, 0, 0], [0, 1, 1, 1]]),
        {'category': ['a', 'b']},
        id='Tracks',
    ),
    pytest.param(
        Vectors,
        np.array([[[0, 0], [1, 1]], [[1, 1], [2, 2]]]),
        {'category': ['a', 'b']},
        id='Vectors',
    ),
]


@pytest.mark.parametrize(('layer_cls', 'data', 'feat'), _LAYER_CASES)
def test_layer_features_polars(layer_cls, data, feat):
    features = pl.DataFrame(feat)
    layer = layer_cls(data, features=features)
    pd.testing.assert_frame_equal(layer.features, pd.DataFrame(feat))
