import numpy as np
import pandas as pd
import pytest
from dask import array as da

from napari.layers.utils.layer_utils import (
    calc_data_range,
    coerce_current_properties,
    dataframe_to_properties,
    dims_displayed_world_to_layer,
    get_current_properties,
    prepare_properties,
    segment_normal,
)

data_dask = da.random.random(
    size=(100_000, 1000, 1000), chunks=(1, 1000, 1000)
)

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


def test_dataframe_to_properties():
    properties = {'point_type': np.array(['A', 'B'] * 5)}
    properties_df = pd.DataFrame(properties)
    converted_properties = dataframe_to_properties(properties_df)
    np.testing.assert_equal(converted_properties, properties)


def test_prepare_properties_with_empty_properties():
    assert prepare_properties({}) == ({}, {})


def test_prepare_properties_with_empty_properties_and_choices():
    assert prepare_properties({}, {}) == ({}, {})


def test_prepare_properties_with_properties_then_choices_from_properties():
    properties, choices = prepare_properties({"aa": [1, 2]}, num_data=2)
    assert list(properties.keys()) == ["aa"]
    assert np.array_equal(properties["aa"], [1, 2])
    assert list(choices.keys()) == ["aa"]
    assert np.array_equal(choices["aa"], [1, 2])


def test_prepare_properties_with_choices_then_properties_are_none():
    properties, choices = prepare_properties({}, {"aa": [1, 2]}, num_data=2)
    assert list(properties.keys()) == ["aa"]
    assert np.array_equal(properties["aa"], [None, None])
    assert list(choices.keys()) == ["aa"]
    assert np.array_equal(choices["aa"], [1, 2])


def test_prepare_properties_with_properties_and_choices():
    properties, choices = prepare_properties({"aa": [1, 2, 1]}, num_data=3)
    assert np.array_equal(properties["aa"], [1, 2, 1])
    assert np.array_equal(choices["aa"], [1, 2])


def test_prepare_properties_with_properties_and_choices_then_merge_choice_values():
    properties, choices = prepare_properties(
        {"aa": [1, 3]}, {"aa": [1, 2]}, num_data=2
    )
    assert list(properties.keys()) == ["aa"]
    assert np.array_equal(properties["aa"], [1, 3])
    assert list(choices.keys()) == ["aa"]
    assert np.array_equal(choices["aa"], [1, 2, 3])


def test_prepare_properties_with_properties_and_choices_then_skip_choice_keys():
    properties, choices = prepare_properties(
        {"aa": [1, 3]}, {"aa": [1, 2], "bb": [7, 6]}, num_data=2
    )
    assert list(properties.keys()) == ["aa"]
    assert np.array_equal(properties["aa"], [1, 3])
    assert list(choices.keys()) == ["aa"]
    assert np.array_equal(choices["aa"], [1, 2, 3])


def test_prepare_properties_with_properties_and_choices_and_save_choices():
    properties, choices = prepare_properties(
        {"aa": [1, 3]},
        {"aa": [1, 2], "bb": [7, 6]},
        num_data=2,
        save_choices=True,
    )
    assert list(properties.keys()) == ["aa", "bb"]
    assert np.array_equal(properties["aa"], [1, 3])
    assert np.array_equal(properties["bb"], [None, None])
    assert list(choices.keys()) == ["aa", "bb"]
    assert np.array_equal(choices["aa"], [1, 2, 3])
    assert np.array_equal(choices["bb"], [6, 7])


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
        properties, property_choices, 0
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
