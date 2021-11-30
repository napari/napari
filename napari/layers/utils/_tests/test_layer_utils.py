import numpy as np
import pandas as pd
import pytest
from dask import array as da

from napari.layers.utils.layer_utils import (
    calc_data_range,
    coerce_current_properties,
    dims_displayed_world_to_layer,
    features_from_properties,
    features_remove,
    features_resize,
    get_current_properties,
    segment_normal,
)

data_dask = da.random.random(
    size=(100_000, 1000, 1000), chunks=(1, 1000, 1000)
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
    assert np.all(clim == [0, 1])

    # all ones should return [0, 1] by default
    data = np.ones((10, 10))
    clim = calc_data_range(data)
    assert np.all(clim == [0, 1])

    # return min and max
    data = np.random.random((10, 15))
    data[0, 0] = 0
    data[0, 1] = 2
    clim = calc_data_range(data)
    assert np.all(clim == [0, 2])

    # return min and max
    data = np.random.random((6, 10, 15))
    data[0, 0, 0] = 0
    data[0, 0, 1] = 2
    clim = calc_data_range(data)
    assert np.all(clim == [0, 2])

    # Try large data
    data = np.zeros((1000, 2000))
    data[0, 0] = 0
    data[0, 1] = 2
    clim = calc_data_range(data)
    assert np.all(clim == [0, 2])

    # Try large data mutlidimensional
    data = np.zeros((3, 1000, 1000))
    data[0, 0, 0] = 0
    data[0, 0, 1] = 2
    clim = calc_data_range(data)
    assert np.all(clim == [0, 2])


def test_calc_data_fast_uint8():
    data = da.random.randint(
        0,
        100,
        size=(1_000, 10, 10),
        chunks=(1, 10, 10),
        dtype=np.uint8,
    )
    assert calc_data_range(data) == [0, 255]


@pytest.mark.timeout(2)
def test_calc_data_range_fast_big():
    val = calc_data_range(data_dask)
    assert len(val) > 0


@pytest.mark.timeout(2)
def test_calc_data_range_fast_big_1d():
    val = calc_data_range(data_dask_1d)
    assert len(val) > 0


@pytest.mark.timeout(2)
def test_calc_data_range_fast_big_1d_rgb():
    val = calc_data_range(data_dask_1d_rgb)
    assert len(val) > 0


@pytest.mark.timeout(2)
def test_calc_data_range_fast_big_plane():
    val = calc_data_range(data_dask_plane)
    assert len(val) > 0


def test_segment_normal_2d():
    a = np.array([1, 1])
    b = np.array([1, 10])

    unit_norm = segment_normal(a, b)
    assert np.all(unit_norm == np.array([1, 0]))


def test_segment_normal_3d():
    a = np.array([1, 1, 0])
    b = np.array([1, 10, 0])
    p = np.array([1, 0, 0])

    unit_norm = segment_normal(a, b, p)
    assert np.all(unit_norm == np.array([0, 0, -1]))


def test_get_current_properties_with_properties_then_last_values():
    properties = {
        "face_color": np.array(["cyan", "red", "red"]),
        "angle": np.array([0.5, 1.5, 1.5]),
    }

    current_properties = get_current_properties(properties, {}, 3)

    assert current_properties == {
        "face_color": "red",
        "angle": 1.5,
    }


def test_get_current_properties_with_property_choices_then_first_values():
    properties = {
        "face_color": np.empty(0, dtype=str),
        "angle": np.empty(0, dtype=float),
    }
    property_choices = {
        "face_color": np.array(["cyan", "red"]),
        "angle": np.array([0.5, 1.5]),
    }

    current_properties = get_current_properties(
        properties,
        property_choices,
    )

    assert current_properties == {
        "face_color": "cyan",
        "angle": 0.5,
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

    for k, v in coerced_current_properties.items():
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
    "dims_displayed,ndim_world,ndim_layer,expected",
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


def test_features_from_properties_with_num_data_only():
    features = features_from_properties(num_data=5)
    assert features.shape == (5, 0)


def test_features_from_properties_with_properties():
    properties = {
        'class': np.array(['sky', 'person', 'building', 'person']),
        'confidence': np.array([0.2, 0.5, 1, 0.8]),
    }

    features = features_from_properties(properties=properties, num_data=4)

    assert features.shape == (4, 2)
    np.testing.assert_array_equal(features['class'], properties['class'])
    np.testing.assert_array_equal(
        features['confidence'], properties['confidence']
    )


def test_features_from_properties_with_properties_and_choices():
    properties = {
        'class': np.array(['sky', 'person', 'building', 'person']),
    }
    property_choices = {
        'class': np.array(['building', 'person', 'sky']),
    }

    features = features_from_properties(
        properties=properties, property_choices=property_choices, num_data=4
    )

    assert features.shape == (4, 1)
    class_column = features['class']
    np.testing.assert_array_equal(class_column, properties['class'])
    assert isinstance(class_column.dtype, pd.CategoricalDtype)
    np.testing.assert_array_equal(
        class_column.dtype.categories, property_choices['class']
    )


def test_features_from_properties_with_choices_only():
    property_choices = {
        'class': np.array(['building', 'person', 'sky']),
    }

    features = features_from_properties(
        property_choices=property_choices, num_data=0
    )

    assert features.shape == (0, 1)
    class_column = features['class']
    assert isinstance(class_column.dtype, pd.CategoricalDtype)
    np.testing.assert_array_equal(
        class_column.dtype.categories, property_choices['class']
    )


def test_from_layer_kwargs_with_empty_properties_and_choices():
    properties = {
        'class': np.array([]),
    }
    property_choices = {
        'class': np.array(['building', 'person', 'sky']),
    }

    features = features_from_properties(
        properties=properties, property_choices=property_choices, num_data=0
    )

    assert features.shape == (0, 1)
    class_column = features['class']
    assert isinstance(class_column.dtype, pd.CategoricalDtype)
    np.testing.assert_array_equal(
        class_column.dtype.categories, property_choices['class']
    )


def test_features_from_properties_with_dataframe():
    properties = pd.DataFrame(
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

    features = features_from_properties(properties=properties)

    pd.testing.assert_frame_equal(features, properties)


def test_features_resize_smaller():
    features = pd.DataFrame(
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
    current_properties = {
        'class': np.array(['person']),
        'confidence': np.array([0.8]),
    }

    new_features = features_resize(features, current_properties, 2)

    assert new_features.shape == (2, 2)
    np.testing.assert_array_equal(new_features['class'], ['sky', 'person'])
    np.testing.assert_array_equal(new_features['confidence'], [0.2, 0.5])


def test_features_resize_larger():
    features = pd.DataFrame(
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
    current_properties = {
        'class': np.array(['person']),
        'confidence': np.array([0.8]),
    }

    new_features = features_resize(features, current_properties, 6)

    assert new_features.shape == (6, 2)
    np.testing.assert_array_equal(
        new_features['class'],
        ['sky', 'person', 'building', 'person', 'person', 'person'],
    )
    np.testing.assert_array_equal(
        new_features['confidence'],
        [0.2, 0.5, 1, 0.8, 0.8, 0.8],
    )


def test_features_remove():
    features = pd.DataFrame(
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

    new_features = features_remove(features, [1, 3])

    assert new_features.shape == (2, 2)
    np.testing.assert_array_equal(
        new_features['class'],
        ['sky', 'building'],
    )
    np.testing.assert_array_equal(
        new_features['confidence'],
        [0.2, 1],
    )
