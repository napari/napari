import sys
import time
from functools import partial
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from dask import array as da
from pretend import stub

from napari.layers import Image
from napari.layers.utils.layer_utils import (
    _determine_class_for_labels,
    _determine_labels_class_based_on_ram,
    _FeatureTable,
    _get_chunk_size,
    _get_tensorstore_or_zarr,
    _get_zeros_for_labels_based_on_module,
    _layers_to_class_set,
    calc_data_range,
    coerce_current_properties,
    dataframe_to_properties,
    dims_displayed_world_to_layer,
    get_current_properties,
    register_layer_attr_action,
    segment_normal,
)
from napari.utils.key_bindings import KeymapHandler, KeymapProvider

data_dask = da.random.random(
    size=(100_000, 1000, 1000), chunks=(1, 1000, 1000)
)
data_dask_8b = da.random.randint(
    0, 100, size=(1_000, 10, 10), chunks=(1, 10, 10), dtype=np.uint8
)
data_dask_1d = da.random.random(size=(20_000_000,), chunks=(5000,))

data_dask_1d_rgb = da.random.random(size=(5_000_000, 3), chunks=(50_000, 3))

data_dask_plane = da.random.random(
    size=(100_000, 100_000), chunks=(1000, 1000)
)


def test_calc_data_range():
    # all zeros should return [0, 1] by default
    data = np.zeros((10, 10))
    clim = calc_data_range(data)
    np.testing.assert_array_equal(clim, (0, 1))

    # all ones should return [0, 1] by default
    data = np.ones((10, 10))
    clim = calc_data_range(data)
    np.testing.assert_array_equal(clim, (0, 1))

    # return min and max
    data = np.random.random((10, 15))
    data[0, 0] = 0
    data[0, 1] = 2
    clim = calc_data_range(data)
    np.testing.assert_array_equal(clim, (0, 2))

    # return min and max
    data = np.random.random((6, 10, 15))
    data[0, 0, 0] = 0
    data[0, 0, 1] = 2
    clim = calc_data_range(data)
    np.testing.assert_array_equal(clim, (0, 2))

    # Try large data
    data = np.zeros((1000, 2000))
    data[0, 0] = 0
    data[0, 1] = 2
    clim = calc_data_range(data)
    np.testing.assert_array_equal(clim, (0, 2))

    # Try large data mutlidimensional
    data = np.zeros((3, 1000, 1000))
    data[0, 0, 0] = 0
    data[0, 0, 1] = 2
    clim = calc_data_range(data)
    np.testing.assert_array_equal(clim, (0, 2))


@pytest.mark.parametrize(
    'data',
    [data_dask_8b, data_dask, data_dask_1d, data_dask_1d_rgb, data_dask_plane],
)
def test_calc_data_range_fast(data):
    now = time.monotonic()
    val = calc_data_range(data)
    assert len(val) > 0
    elapsed = time.monotonic() - now
    assert elapsed < 5, 'test took too long, computation was likely not lazy'


def test_segment_normal_2d():
    a = np.array([1, 1])
    b = np.array([1, 10])

    unit_norm = segment_normal(a, b)
    np.testing.assert_array_equal(unit_norm, np.array([1, 0]))


def test_segment_normal_3d():
    a = np.array([1, 1, 0])
    b = np.array([1, 10, 0])
    p = np.array([1, 0, 0])

    unit_norm = segment_normal(a, b, p)
    np.testing.assert_array_equal(unit_norm, np.array([0, 0, -1]))


def test_dataframe_to_properties():
    properties = {'point_type': np.array(['A', 'B'] * 5)}
    properties_df = pd.DataFrame(properties)
    converted_properties = dataframe_to_properties(properties_df)
    np.testing.assert_equal(converted_properties, properties)


def test_get_current_properties_with_properties_then_last_values():
    properties = {
        'face_color': np.array(['cyan', 'red', 'red']),
        'angle': np.array([0.5, 1.5, 1.5]),
    }

    current_properties = get_current_properties(properties, {}, 3)

    assert current_properties == {
        'face_color': 'red',
        'angle': 1.5,
    }


def test_get_current_properties_with_property_choices_then_first_values():
    properties = {
        'face_color': np.empty(0, dtype=str),
        'angle': np.empty(0, dtype=float),
    }
    property_choices = {
        'face_color': np.array(['cyan', 'red']),
        'angle': np.array([0.5, 1.5]),
    }

    current_properties = get_current_properties(
        properties,
        property_choices,
    )

    assert current_properties == {
        'face_color': 'cyan',
        'angle': 0.5,
    }


def test_coerce_current_properties_valid_values():
    current_properties = {
        'annotation': ['leg'],
        'confidence': 1,
        'annotator': 'ash',
        'model': np.array(['best']),
    }
    expected_current_properties = {
        'annotation': np.array(['leg']),
        'confidence': np.array([1]),
        'annotator': np.array(['ash']),
        'model': np.array(['best']),
    }
    coerced_current_properties = coerce_current_properties(current_properties)

    for k in coerced_current_properties:
        value = coerced_current_properties[k]
        assert isinstance(value, np.ndarray)
        np.testing.assert_equal(value, expected_current_properties[k])


def test_coerce_current_properties_invalid_values():
    current_properties = {
        'annotation': ['leg'],
        'confidence': 1,
        'annotator': 'ash',
        'model': np.array(['best', 'best_v2_final']),
    }

    with pytest.raises(ValueError):
        _ = coerce_current_properties(current_properties)


@pytest.mark.parametrize(
    'dims_displayed,ndim_world,ndim_layer,expected',
    [
        ([1, 2, 3], 4, 4, [1, 2, 3]),
        ([0, 1, 2], 4, 4, [0, 1, 2]),
        ([1, 2, 3], 4, 3, [0, 1, 2]),
        ([0, 1, 2], 4, 3, [2, 0, 1]),
        ([1, 2, 3], 4, 2, [0, 1]),
        ([0, 1, 2], 3, 3, [0, 1, 2]),
        ([0, 1], 2, 2, [0, 1]),
        ([1, 0], 2, 2, [1, 0]),
    ],
)
def test_dims_displayed_world_to_layer(
    dims_displayed, ndim_world, ndim_layer, expected
):
    dims_displayed_layer = dims_displayed_world_to_layer(
        dims_displayed, ndim_world=ndim_world, ndim_layer=ndim_layer
    )
    np.testing.assert_array_equal(dims_displayed_layer, expected)


def test_feature_table_from_layer_with_none_then_empty():
    feature_table = _FeatureTable.from_layer(features=None)
    assert feature_table.values.shape == (0, 0)


def test_feature_table_from_layer_with_num_data_only():
    feature_table = _FeatureTable.from_layer(num_data=5)
    assert feature_table.values.shape == (5, 0)
    assert feature_table.defaults.shape == (1, 0)


def test_feature_table_from_layer_with_empty_int_features():
    feature_table = _FeatureTable.from_layer(
        features={'a': np.empty(0, dtype=np.int64)}
    )
    assert feature_table.values['a'].dtype == np.int64
    assert len(feature_table.values['a']) == 0
    assert feature_table.defaults['a'].dtype == np.int64
    assert feature_table.defaults['a'][0] == 0


def test_feature_table_from_layer_with_properties_and_num_data():
    properties = {
        'class': np.array(['sky', 'person', 'building', 'person']),
        'confidence': np.array([0.2, 0.5, 1, 0.8]),
        'varying_length_prop': np.array(
            [[0], [0, 0, 0], [0, 0], [0]], dtype=object
        ),
    }

    feature_table = _FeatureTable.from_layer(properties=properties, num_data=4)

    features = feature_table.values
    assert features.shape == (4, 3)
    np.testing.assert_array_equal(features['class'], properties['class'])
    np.testing.assert_array_equal(
        features['confidence'], properties['confidence']
    )
    np.testing.assert_array_equal(
        features['varying_length_prop'], properties['varying_length_prop']
    )

    defaults = feature_table.defaults
    assert defaults.shape == (1, 3)
    assert defaults['class'][0] == properties['class'][-1]
    assert defaults['confidence'][0] == properties['confidence'][-1]
    assert (
        defaults['varying_length_prop'][0]
        == properties['varying_length_prop'][-1]
    )


def test_feature_table_from_layer_with_properties_and_choices():
    properties = {
        'class': np.array(['sky', 'person', 'building', 'person']),
    }
    property_choices = {
        'class': np.array(['building', 'person', 'sky']),
    }

    feature_table = _FeatureTable.from_layer(
        properties=properties, property_choices=property_choices, num_data=4
    )

    features = feature_table.values
    assert features.shape == (4, 1)
    class_column = features['class']
    np.testing.assert_array_equal(class_column, properties['class'])
    assert isinstance(class_column.dtype, pd.CategoricalDtype)
    np.testing.assert_array_equal(
        class_column.dtype.categories, property_choices['class']
    )
    defaults = feature_table.defaults
    assert defaults.shape == (1, 1)
    assert defaults['class'][0] == properties['class'][-1]


def test_feature_table_from_layer_with_choices_only():
    property_choices = {
        'class': np.array(['building', 'person', 'sky']),
    }

    feature_table = _FeatureTable.from_layer(
        property_choices=property_choices, num_data=0
    )

    features = feature_table.values
    assert features.shape == (0, 1)
    class_column = features['class']
    assert isinstance(class_column.dtype, pd.CategoricalDtype)
    np.testing.assert_array_equal(
        class_column.dtype.categories, property_choices['class']
    )
    defaults = feature_table.defaults
    assert defaults.shape == (1, 1)
    assert defaults['class'][0] == property_choices['class'][0]


def test_feature_table_from_layer_with_empty_properties_and_choices():
    properties = {
        'class': np.array([]),
    }
    property_choices = {
        'class': np.array(['building', 'person', 'sky']),
    }

    feature_table = _FeatureTable.from_layer(
        properties=properties, property_choices=property_choices, num_data=0
    )

    features = feature_table.values
    assert features.shape == (0, 1)
    class_column = features['class']
    assert isinstance(class_column.dtype, pd.CategoricalDtype)
    np.testing.assert_array_equal(
        class_column.dtype.categories, property_choices['class']
    )
    defaults = feature_table.defaults
    assert defaults.shape == (1, 1)
    assert defaults['class'][0] == property_choices['class'][0]


TEST_FEATURES = pd.DataFrame(
    {
        'class': pd.Series(
            ['sky', 'person', 'building', 'person'],
            dtype=pd.CategoricalDtype(
                categories=('building', 'person', 'sky')
            ),
        ),
        'confidence': pd.Series([0.2, 0.5, 1, 0.8]),
    }
)


def test_feature_table_from_layer_with_properties_as_dataframe():
    feature_table = _FeatureTable.from_layer(properties=TEST_FEATURES)
    pd.testing.assert_frame_equal(feature_table.values, TEST_FEATURES)


@pytest.fixture
def feature_table():
    return _FeatureTable(TEST_FEATURES.copy(deep=True), num_data=4)


def test_feature_table_resize_smaller(feature_table: _FeatureTable):
    feature_table.resize(2)

    features = feature_table.values
    assert features.shape == (2, 2)
    np.testing.assert_array_equal(features['class'], ['sky', 'person'])
    np.testing.assert_array_equal(features['confidence'], [0.2, 0.5])


def test_feature_table_resize_larger(feature_table: _FeatureTable):
    expected_dtypes = feature_table.values.dtypes

    feature_table.resize(6)

    features = feature_table.values
    assert features.shape == (6, 2)
    np.testing.assert_array_equal(
        features['class'],
        ['sky', 'person', 'building', 'person', 'person', 'person'],
    )
    np.testing.assert_array_equal(
        features['confidence'],
        [0.2, 0.5, 1, 0.8, 0.8, 0.8],
    )
    np.testing.assert_array_equal(features.dtypes, expected_dtypes)


def test_feature_table_append(feature_table: _FeatureTable):
    to_append = pd.DataFrame(
        {
            'class': ['sky', 'building'],
            'confidence': [0.6, 0.1],
        }
    )

    feature_table.append(to_append)

    features = feature_table.values
    assert features.shape == (6, 2)
    np.testing.assert_array_equal(
        features['class'],
        ['sky', 'person', 'building', 'person', 'sky', 'building'],
    )
    np.testing.assert_array_equal(
        features['confidence'],
        [0.2, 0.5, 1, 0.8, 0.6, 0.1],
    )


def test_feature_table_remove(feature_table: _FeatureTable):
    feature_table.remove([1, 3])

    features = feature_table.values
    assert features.shape == (2, 2)
    np.testing.assert_array_equal(features['class'], ['sky', 'building'])
    np.testing.assert_array_equal(features['confidence'], [0.2, 1])


def test_feature_table_from_layer_with_custom_index():
    features = pd.DataFrame({'a': [1, 3], 'b': [7.5, -2.1]}, index=[1, 2])
    feature_table = _FeatureTable.from_layer(features=features)
    expected = features.reset_index(drop=True)
    pd.testing.assert_frame_equal(feature_table.values, expected)


def test_feature_table_from_layer_with_custom_index_and_num_data():
    features = pd.DataFrame({'a': [1, 3], 'b': [7.5, -2.1]}, index=[1, 2])
    feature_table = _FeatureTable.from_layer(features=features, num_data=2)
    expected = features.reset_index(drop=True)
    pd.testing.assert_frame_equal(feature_table.values, expected)


def test_feature_table_from_layer_with_unordered_pd_series_properties():
    properties = {
        'a': pd.Series([1, 3], index=[3, 4]),
        'b': pd.Series([7.5, -2.1], index=[1, 2]),
    }
    feature_table = _FeatureTable.from_layer(properties=properties, num_data=2)
    expected = pd.DataFrame({'a': [1, 3], 'b': [7.5, -2.1]}, index=[0, 1])
    pd.testing.assert_frame_equal(feature_table.values, expected)


def test_feature_table_from_layer_with_unordered_pd_series_features():
    features = {
        'a': pd.Series([1, 3], index=[3, 4]),
        'b': pd.Series([7.5, -2.1], index=[1, 2]),
    }
    feature_table = _FeatureTable.from_layer(features=features, num_data=2)
    expected = pd.DataFrame({'a': [1, 3], 'b': [7.5, -2.1]}, index=[0, 1])
    pd.testing.assert_frame_equal(feature_table.values, expected)


def test_feature_table_set_defaults_with_same_columns(feature_table):
    defaults = {'class': 'building', 'confidence': 1}
    assert feature_table.defaults['class'][0] != defaults['class']
    assert feature_table.defaults['confidence'][0] != defaults['confidence']

    feature_table.set_defaults(defaults)

    assert feature_table.defaults['class'][0] == defaults['class']
    assert feature_table.defaults['confidence'][0] == defaults['confidence']


def test_feature_table_set_defaults_with_extra_column(feature_table):
    defaults = {'class': 'building', 'confidence': 0, 'cat': 'kermit'}
    assert 'cat' not in feature_table.values.columns
    with pytest.raises(ValueError):
        feature_table.set_defaults(defaults)


def test_feature_table_set_defaults_with_missing_column(feature_table):
    defaults = {'class': 'building'}
    assert len(feature_table.values.columns) > 1
    with pytest.raises(ValueError):
        feature_table.set_defaults(defaults)


def test_register_label_attr_action(monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 1)

    class Foo(KeymapProvider):
        def __init__(self) -> None:
            super().__init__()
            self.value = 0

    foo = Foo()

    handler = KeymapHandler()
    handler.keymap_providers = [foo]

    @register_layer_attr_action(Foo, 'value desc', 'value', 'K')
    def set_value_1(x):
        x.value = 1

    handler.press_key('K')
    assert foo.value == 1
    handler.release_key('K')
    assert foo.value == 1

    foo.value = 0
    handler.press_key('K')
    assert foo.value == 1
    monkeypatch.setattr(time, 'time', lambda: 2)
    handler.release_key('K')
    assert foo.value == 0


def test_numpy_chunk_size():
    assert _get_chunk_size(np.zeros((100, 100))) is None
    assert _get_chunk_size(list(range(10))) is None


def test_zarr_get_chunk_size():
    import zarr

    data_shape = (100, 100)
    chunk_shape = (10, 10)

    data = zarr.zeros(data_shape, chunks=chunk_shape, dtype='u2')
    chunk_size = _get_chunk_size(data)
    assert np.array_equal(chunk_size, chunk_shape)


def test_xarray_get_chunk_size():
    import xarray as xr

    data_shape = (100, 100)
    chunk_shape = (10, 10)
    chunk_shape_dict = {'x': 10, 'y': 10}

    coords = list(range(100))
    data = xr.DataArray(
        np.zeros(data_shape),
        dims=['y', 'x'],
        coords={'y': coords, 'x': coords},
    )
    data = data.chunk(chunk_shape_dict)
    chunk_size = _get_chunk_size(data)
    assert np.array_equal(chunk_size, chunk_shape)


def test_dask_get_chunk_size():
    chunk_size = _get_chunk_size(data_dask_plane)
    assert np.array_equal(chunk_size, (1000, 1000))


def test_tensorstore_get_chunk_size():
    ts = pytest.importorskip('tensorstore')

    data_shape = (100, 100)
    chunk_shape = (10, 10)

    spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'memory'},
        'metadata': {'chunks': chunk_shape},
    }
    labels = ts.open(
        spec, create=True, dtype='uint32', shape=data_shape
    ).result()

    chunk_size = _get_chunk_size(labels)
    assert np.array_equal(chunk_size, chunk_shape)


def test_determine_class_for_labels(monkeypatch):
    monkeypatch.delitem(sys.modules, 'tensorstore', raising=False)
    assert _determine_class_for_labels(set()) == np.ndarray
    assert _determine_class_for_labels({np.ndarray}) == np.ndarray
    assert _determine_class_for_labels({np.ndarray, da.Array}) == zarr.Array
    assert _determine_class_for_labels({da.Array}) == zarr.Array


def test_determine_class_for_labels_tensorstore():
    ts = pytest.importorskip('tensorstore')
    assert (
        _determine_class_for_labels({np.ndarray, da.Array}) == ts.TensorStore
    )
    assert _determine_class_for_labels({np.ndarray, zarr.Array}) == zarr.Array
    assert (
        _determine_class_for_labels({np.ndarray, zarr.Array, ts.TensorStore})
        == ts.TensorStore
    )


def test_determine_class_for_labels_warning(monkeypatch):
    monkeypatch.delitem(sys.modules, 'tensorstore', raising=False)
    monkeypatch.delitem(sys.modules, 'zarr', raising=False)

    with pytest.warns(RuntimeWarning, match='We cannot use dask.array'):
        assert _determine_class_for_labels({da.Array}) == np.ndarray


def test_get_zeros_for_labels_based_on_module(monkeypatch):
    assert _get_zeros_for_labels_based_on_module(None, None) is np.zeros
    assert _get_zeros_for_labels_based_on_module(np, None) is np.zeros
    assert _get_zeros_for_labels_based_on_module(zarr, None) is zarr.zeros
    res = _get_zeros_for_labels_based_on_module(zarr, (10, 10))
    assert isinstance(res, partial)
    assert res.keywords == {'chunks': (10, 10)}
    assert res.func is zarr.zeros


def test_get_zeros_for_labels_based_on_module_dask(monkeypatch):
    def _get_zarr():
        return zarr

    monkeypatch.setattr(
        'napari.layers.utils.layer_utils._get_tensorstore_or_zarr', _get_zarr
    )
    assert _get_zeros_for_labels_based_on_module(da, None) is zarr.zeros


def test_get_zeros_for_labels_based_on_module_dask_warning(monkeypatch):
    def _get_zarr():
        return None

    monkeypatch.setattr(
        'napari.layers.utils.layer_utils._get_tensorstore_or_zarr', _get_zarr
    )
    with pytest.warns(RuntimeWarning, match='We cannot use dask.array'):
        assert _get_zeros_for_labels_based_on_module(da, None) is np.zeros


def test_get_zeros_for_labels_based_on_module_dask_tensorstore(monkeypatch):
    ts = pytest.importorskip('tensorstore')

    def _get_ts():
        return ts

    monkeypatch.setattr(
        'napari.layers.utils.layer_utils._get_tensorstore_or_zarr', _get_ts
    )
    res = _get_zeros_for_labels_based_on_module(da, (10, 10))

    arr = res(shape=(100, 100), dtype='u2')

    assert isinstance(arr, ts.TensorStore)
    assert arr.shape == (100, 100)
    assert arr.chunk_layout.read_chunk.shape == (10, 10)


def test_get_zeros_for_labels_based_on_module_unknown():
    with pytest.warns(RuntimeWarning, match='Unknown data library'):
        assert _get_zeros_for_labels_based_on_module(pytest, None) is np.zeros


def test_get_zeros_for_labels_based_on_module_tensorstore():
    ts = pytest.importorskip('tensorstore')

    res = _get_zeros_for_labels_based_on_module(ts, (10, 10))

    arr = res(shape=(100, 100), dtype='u2')

    assert isinstance(arr, ts.TensorStore)
    assert arr.shape == (100, 100)
    assert arr.chunk_layout.read_chunk.shape == (10, 10)


def test_layers_to_class_set():
    d1 = Image(np.zeros((10, 10)))
    d2 = Image(np.zeros((10, 10)))
    d3 = Image(zarr.zeros((10, 10)))
    d4 = Image(xr.DataArray(np.zeros((10, 10)), dims=['x', 'y']))
    d5 = Image(d4.data.chunk({'x': 5, 'y': 5}))
    assert _layers_to_class_set([d1, d2]) == {np.ndarray}
    assert _layers_to_class_set([d1, d2, d3]) == {np.ndarray, zarr.Array}
    assert _layers_to_class_set([d4]) == {np.ndarray}
    assert _layers_to_class_set([d5]) == {da.Array}


@pytest.mark.parametrize('module_name', ['numpy', 'zarr.core', 'tensorstore'])
@patch(
    'psutil.virtual_memory',
    return_value=stub(total=1000000000, available=1000000000),
)
def test_determine_labels_class_based_on_ram(virtual_memory, module_name):
    module = pytest.importorskip(module_name)

    zeros = _get_zeros_for_labels_based_on_module(module, (10, 10))(
        shape=(100, 100), dtype='uint8'
    )

    assert (
        _determine_labels_class_based_on_ram(
            zeros.__class__, zeros.shape, 'uint8'
        )
        == module
    )

    virtual_memory.return_value = stub(total=1000000000, available=10)

    assert (
        _determine_labels_class_based_on_ram(
            zeros.__class__, zeros.shape, 'uint8'
        )
        == _get_tensorstore_or_zarr()
    )


@patch(
    'psutil.virtual_memory',
    return_value=stub(total=1000000000, available=1000000000),
)
def test_determine_labels_class_based_on_ram_dask(virtual_memory):
    zeros = da.zeros((100, 100), chunks=(10, 10), dtype='uint8')

    assert (
        _determine_labels_class_based_on_ram(
            zeros.__class__, zeros.shape, 'uint8'
        )
        == da.core
    )
