"""Tests verifying eager narwhals-compatible DataFrames are accepted.

Covers every feature/property boundary function that now accepts
``IntoDataFrame`` instead of only ``pd.DataFrame``.
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
# Backend-agnostic — each test runs for both pandas and polars
# ---------------------------------------------------------------------------

_BACKENDS = [
    pytest.param(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), id='pandas'),
    pytest.param(pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), id='polars'),
]


@pytest.fixture(params=_BACKENDS)
def any_df(request):
    return request.param


@pytest.mark.parametrize(('func', 'expected'), [
    pytest.param(
        lambda df: (_validate_features(df),),
        lambda r: (isinstance(r, pd.DataFrame) and list(r.columns) == ['a', 'b']),
        id='validate_features',
    ),
    pytest.param(
        lambda df: (_FeatureTable(df),),
        lambda r: isinstance(r.values, pd.DataFrame),
        id='feature_table',
    ),
    pytest.param(
        lambda df: (dataframe_to_properties(df),),
        lambda r: isinstance(r, dict) and 'a' in r,
        id='dataframe_to_properties',
    ),
    pytest.param(
        lambda df: (validate_properties(df),),
        lambda r: isinstance(r, dict),
        id='validate_properties',
    ),
])
def test_backend_agnostic(any_df, func, expected):
    result, = func(any_df)
    assert expected(result)


# ---------------------------------------------------------------------------
# dict / None inputs (not backend-specific)
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
# Polars-specific — code paths that only activate with a non-pandas input
# ---------------------------------------------------------------------------

def test_validate_feature_defaults_polars():
    """_validate_feature_defaults accepts a Polars DataFrame (triggers
    the _to_pandas conversion inside the non-dict branch)."""
    ft = _FeatureTable(pl.DataFrame({'x': [10, 20, 30], 'y': ['a', 'b', 'c']}))
    defaults = _validate_feature_defaults(
        pl.DataFrame({'x': [99], 'y': ['z']}), ft.values
    )
    assert isinstance(defaults, pd.DataFrame)


def test_features_from_properties_polars():
    """_features_from_properties accepts a Polars DataFrame when
    property_choices are supplied (triggers the non-dict conversion)."""
    result = _features_from_properties(
        properties=pl.DataFrame({'x': [10, 20, 30], 'y': ['a', 'b', 'c']}),
        property_choices={'x': np.array([10, 20, 30])},
    )
    assert isinstance(result, pd.DataFrame)


def test_to_pandas():
    plf = pl.DataFrame({'x': [10, 20, 30], 'y': ['a', 'b', 'c']})
    result = _to_pandas(plf)
    assert isinstance(result, pd.DataFrame)


def test_to_pandas_without_pyarrow(monkeypatch):
    import narwhals as nw

    plf = pl.DataFrame({'x': [10, 20, 30], 'y': ['a', 'b', 'c']})

    def raise_import_error(_):
        raise ImportError

    monkeypatch.setattr(nw.DataFrame, 'to_pandas', raise_import_error)
    result = _to_pandas(plf)
    pd.testing.assert_frame_equal(
        result, pd.DataFrame({'x': [10, 20, 30], 'y': ['a', 'b', 'c']})
    )


# ---------------------------------------------------------------------------
# Layer integration — each concrete layer accepts Polars features
# ---------------------------------------------------------------------------

_LAYER_CASES = [
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
]


@pytest.mark.parametrize(('layer_cls', 'data', 'feat'), _LAYER_CASES)
def test_layer_features_polars(layer_cls, data, feat):
    features = pl.DataFrame(feat)
    layer = layer_cls(data, features=features)
    pd.testing.assert_frame_equal(layer.features, pd.DataFrame(feat))
