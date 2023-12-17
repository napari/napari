import importlib
from itertools import product
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest

from napari.utils.color import ColorArray
from napari.utils.colormaps import Colormap, colormap
from napari.utils.colormaps.colormap import (
    MAPPING_OF_UNKNOWN_VALUE,
    DirectLabelColormap,
)
from napari.utils.colormaps.colormap_utils import label_colormap


def test_linear_colormap():
    """Test a linear colormap."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap = Colormap(colors, name='testing')

    assert cmap.name == 'testing'
    assert cmap.interpolation == 'linear'
    assert len(cmap.controls) == len(colors)
    np.testing.assert_almost_equal(cmap.colors, colors)
    np.testing.assert_almost_equal(cmap.map([0.75]), [[0, 0.5, 0.5, 1]])


def test_linear_colormap_with_control_points():
    """Test a linear colormap with control points."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap = Colormap(colors, name='testing', controls=[0, 0.75, 1])

    assert cmap.name == 'testing'
    assert cmap.interpolation == 'linear'
    assert len(cmap.controls) == len(colors)
    np.testing.assert_almost_equal(cmap.colors, colors)
    np.testing.assert_almost_equal(cmap.map([0.75]), [[0, 1, 0, 1]])


def test_non_ascending_control_points():
    """Test non ascending control points raises an error."""
    colors = np.array(
        [[0, 0, 0, 1], [0, 0.5, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    )
    with pytest.raises(ValueError):
        Colormap(colors, name='testing', controls=[0, 0.75, 0.25, 1])


def test_wrong_number_control_points():
    """Test wrong number of control points raises an error."""
    colors = np.array(
        [[0, 0, 0, 1], [0, 0.5, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    )
    with pytest.raises(ValueError):
        Colormap(colors, name='testing', controls=[0, 0.75, 1])


def test_wrong_start_control_point():
    """Test wrong start of control points raises an error."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    with pytest.raises(ValueError):
        Colormap(colors, name='testing', controls=[0.1, 0.75, 1])


def test_wrong_end_control_point():
    """Test wrong end of control points raises an error."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    with pytest.raises(ValueError):
        Colormap(colors, name='testing', controls=[0, 0.75, 0.9])


def test_binned_colormap():
    """Test a binned colormap."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap = Colormap(colors, name='testing', interpolation='zero')

    assert cmap.name == 'testing'
    assert cmap.interpolation == 'zero'
    assert len(cmap.controls) == len(colors) + 1
    np.testing.assert_almost_equal(cmap.colors, colors)
    np.testing.assert_almost_equal(cmap.map([0.4]), [[0, 1, 0, 1]])


def test_binned_colormap_with_control_points():
    """Test a binned with control points."""
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap = Colormap(
        colors,
        name='testing',
        interpolation='zero',
        controls=[0, 0.2, 0.3, 1],
    )

    assert cmap.name == 'testing'
    assert cmap.interpolation == 'zero'
    assert len(cmap.controls) == len(colors) + 1
    np.testing.assert_almost_equal(cmap.colors, colors)
    np.testing.assert_almost_equal(cmap.map([0.4]), [[0, 0, 1, 1]])


def test_colormap_equality():
    colors = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    cmap_1 = Colormap(colors, name='testing', controls=[0, 0.75, 1])
    cmap_2 = Colormap(colors, name='testing', controls=[0, 0.75, 1])
    cmap_3 = Colormap(colors, name='testing', controls=[0, 0.25, 1])
    assert cmap_1 == cmap_2
    assert cmap_1 != cmap_3


def test_colormap_recreate():
    c_map = Colormap("black")
    Colormap(**c_map.dict())


@pytest.mark.parametrize('ndim', range(1, 5))
def test_mapped_shape(ndim):
    np.random.seed(0)
    img = np.random.random((5,) * ndim)
    cmap = Colormap(colors=['red'])
    mapped = cmap.map(img)
    assert mapped.shape == img.shape + (4,)


@pytest.mark.parametrize(
    "num,dtype", [(40, np.uint8), (1000, np.uint16), (80000, np.float32)]
)
def test_minimum_dtype_for_labels(num, dtype):
    assert colormap.minimum_dtype_for_labels(num) == dtype


@pytest.fixture()
def disable_jit(monkeypatch):
    pytest.importorskip("numba")
    with patch("numba.core.config.DISABLE_JIT", True):
        importlib.reload(colormap)
        yield
    importlib.reload(colormap)  # revert to original state


@pytest.mark.parametrize(
    "num,dtype", [(40, np.uint8), (1000, np.uint16), (80000, np.float32)]
)
@pytest.mark.usefixtures("disable_jit")
def test_cast_labels_to_minimum_type_auto(num: int, dtype, monkeypatch):
    cmap = label_colormap(num)
    data = np.zeros(3, dtype=np.uint32)
    data[1] = 10
    data[2] = 10**6 + 5
    cast_arr = colormap._cast_labels_data_to_texture_dtype_auto(data, cmap)
    assert cast_arr.dtype == dtype
    assert cast_arr[0] == 0
    assert cast_arr[1] == 10
    assert cast_arr[2] == 10**6 % num + 5


@pytest.fixture
def direct_label_colormap():
    return DirectLabelColormap(
        color_dict={
            0: np.array([0, 0, 0, 0]),
            1: np.array([255, 0, 0, 255]),
            2: np.array([0, 255, 0, 255]),
            3: np.array([0, 0, 255, 255]),
            12: np.array([0, 0, 255, 255]),
            None: np.array([255, 255, 255, 255]),
        },
    )


def test_direct_label_colormap_simple(direct_label_colormap):
    np.testing.assert_array_equal(
        direct_label_colormap.map([0, 2, 7]),
        np.array([[0, 0, 0, 0], [0, 255, 0, 255], [255, 255, 255, 255]]),
    )
    assert direct_label_colormap._num_unique_colors == 5

    (
        label_mapping,
        color_dict,
    ) = direct_label_colormap._values_mapping_to_minimum_values_set()

    assert len(label_mapping) == 6
    assert len(color_dict) == 5
    assert label_mapping[None] == MAPPING_OF_UNKNOWN_VALUE
    assert label_mapping[12] == label_mapping[3]
    np.testing.assert_array_equal(
        color_dict[label_mapping[0]], direct_label_colormap.color_dict[0]
    )
    np.testing.assert_array_equal(
        color_dict[0], direct_label_colormap.color_dict[None]
    )


def test_direct_label_colormap_selection(direct_label_colormap):
    direct_label_colormap.selection = 2
    direct_label_colormap.use_selection = True

    np.testing.assert_array_equal(
        direct_label_colormap.map([0, 2, 7]),
        np.array([[0, 0, 0, 0], [0, 255, 0, 255], [0, 0, 0, 0]]),
    )

    (
        label_mapping,
        color_dict,
    ) = direct_label_colormap._values_mapping_to_minimum_values_set()

    assert len(label_mapping) == 2
    assert len(color_dict) == 2


@pytest.mark.usefixtures("disable_jit")
def test_cast_direct_labels_to_minimum_type(direct_label_colormap):
    data = np.arange(15, dtype=np.uint32)
    cast = colormap._labels_raw_to_texture_direct(data, direct_label_colormap)
    label_mapping = (
        direct_label_colormap._values_mapping_to_minimum_values_set()[0]
    )
    assert cast.dtype == np.uint8
    np.testing.assert_array_equal(
        cast,
        np.array(
            [
                label_mapping[0],
                label_mapping[1],
                label_mapping[2],
                label_mapping[3],
                MAPPING_OF_UNKNOWN_VALUE,
                MAPPING_OF_UNKNOWN_VALUE,
                MAPPING_OF_UNKNOWN_VALUE,
                MAPPING_OF_UNKNOWN_VALUE,
                MAPPING_OF_UNKNOWN_VALUE,
                MAPPING_OF_UNKNOWN_VALUE,
                MAPPING_OF_UNKNOWN_VALUE,
                MAPPING_OF_UNKNOWN_VALUE,
                label_mapping[3],
                MAPPING_OF_UNKNOWN_VALUE,
                MAPPING_OF_UNKNOWN_VALUE,
            ]
        ),
    )


@pytest.mark.parametrize(
    "num,dtype", [(40, np.uint8), (1000, np.uint16), (80000, np.float32)]
)
@pytest.mark.usefixtures("disable_jit")
def test_test_cast_direct_labels_to_minimum_type_no_jit(num, dtype):
    cmap = DirectLabelColormap(
        color_dict={
            k: np.array([*v, 255])
            for k, v in zip(range(num), product(range(256), repeat=3))
        },
    )
    cmap.color_dict[None] = np.array([255, 255, 255, 255])
    data = np.arange(10, dtype=np.uint32)
    data[2] = 80005
    cast = colormap._labels_raw_to_texture_direct(data, cmap)
    assert cast.dtype == dtype


def test_zero_preserving_modulo_naive():
    pytest.importorskip("numba")
    data = np.arange(1000, dtype=np.uint32)
    res1 = colormap._zero_preserving_modulo_numpy(data, 49, np.uint8)
    res2 = colormap._zero_preserving_modulo(data, 49, np.uint8)
    npt.assert_array_equal(res1, res2)


@pytest.mark.parametrize(
    'dtype', [np.uint8, np.uint16, np.int8, np.int16, np.float32, np.float64]
)
def test_label_colormap_map_with_uint8_values(dtype):
    cmap = colormap.LabelColormap(
        colors=ColorArray(np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]]))
    )
    values = np.array([0, 1, 2], dtype=dtype)
    expected = np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
    npt.assert_array_equal(cmap.map(values), expected)


@pytest.mark.parametrize("selection", [1, -1])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_label_colormap_map_with_selection(selection, dtype):
    cmap = colormap.LabelColormap(
        colors=ColorArray(
            np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
        ),
        use_selection=True,
        selection=selection,
    )
    values = np.array([0, selection, 2], dtype=np.int8)
    expected = np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]])
    npt.assert_array_equal(cmap.map(values), expected)


@pytest.mark.parametrize("background", [1, -1])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_label_colormap_map_with_background(background, dtype):
    cmap = colormap.LabelColormap(
        colors=ColorArray(
            np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
        ),
        background_value=background,
    )
    values = np.array([3, background, 2], dtype=dtype)
    expected = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]])
    npt.assert_array_equal(cmap.map(values), expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_label_colormap_using_cache(dtype, monkeypatch):
    cmap = colormap.LabelColormap(
        colors=ColorArray(np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]]))
    )
    values = np.array([0, 1, 2], dtype=dtype)
    expected = np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
    map1 = cmap.map(values)
    npt.assert_array_equal(map1, expected)
    monkeypatch.setattr(colormap, '_zero_preserving_modulo_numpy', None)
    map2 = cmap.map(values)
    npt.assert_array_equal(map1, map2)


@pytest.mark.parametrize("size", [100, 1000])
def test_cast_direct_labels_to_minimum_type_naive(size):
    pytest.importorskip("numba")
    data = np.arange(size, dtype=np.uint32)
    dtype = colormap.minimum_dtype_for_labels(size)
    cmap = DirectLabelColormap(
        color_dict={
            k: np.array([*v, 255])
            for k, v in zip(range(size - 2), product(range(256), repeat=3))
        },
    )
    cmap.color_dict[None] = np.array([255, 255, 255, 255])
    res1 = colormap._labels_raw_to_texture_direct(data, cmap)
    res2 = colormap._labels_raw_to_texture_direct_numpy(data, cmap)
    npt.assert_array_equal(res1, res2)
    assert res1.dtype == dtype
    assert res2.dtype == dtype


def test_direct_colormap_with_no_selection():
    # Create a DirectLabelColormap with a simple color_dict
    color_dict = {1: np.array([1, 0, 0, 1]), 2: np.array([0, 1, 0, 1])}
    cmap = DirectLabelColormap(color_dict=color_dict)

    # Map a single value
    mapped = cmap.map(1)
    npt.assert_array_equal(mapped[0], np.array([1, 0, 0, 1]))

    # Map multiple values
    mapped = cmap.map(np.array([1, 2]))
    npt.assert_array_equal(mapped, np.array([[1, 0, 0, 1], [0, 1, 0, 1]]))


def test_direct_colormap_with_selection():
    # Create a DirectLabelColormap with a simple color_dict and a selection
    color_dict = {1: np.array([1, 0, 0, 1]), 2: np.array([0, 1, 0, 1])}
    cmap = DirectLabelColormap(
        color_dict=color_dict, use_selection=True, selection=1
    )

    # Map a single value
    mapped = cmap.map(1)
    npt.assert_array_equal(mapped[0], np.array([1, 0, 0, 1]))

    # Map a value that is not the selection
    mapped = cmap.map(2)
    npt.assert_array_equal(mapped[0], np.array([0, 0, 0, 0]))


def test_direct_colormap_with_invalid_values():
    # Create a DirectLabelColormap with a simple color_dict
    color_dict = {1: np.array([1, 0, 0, 1]), 2: np.array([0, 1, 0, 1])}
    cmap = DirectLabelColormap(color_dict=color_dict)

    # Map a value that is not in the color_dict
    mapped = cmap.map(3)
    npt.assert_array_equal(mapped[0], np.array([0, 0, 0, 0]))


def test_direct_colormap_with_empty_color_dict():
    # Create a DirectLabelColormap with an empty color_dict
    cmap = DirectLabelColormap(color_dict={})

    # Map a value
    mapped = cmap.map(1)
    npt.assert_array_equal(mapped[0], np.array([0, 0, 0, 0]))


def test_direct_colormap_with_non_integer_values():
    # Create a DirectLabelColormap with a simple color_dict
    color_dict = {1: np.array([1, 0, 0, 1]), 2: np.array([0, 1, 0, 1])}
    cmap = DirectLabelColormap(color_dict=color_dict)

    # Map a float value
    with pytest.raises(TypeError, match='DirectLabelColormap can only'):
        cmap.map(1.5)

    # Map a string value
    with pytest.raises(TypeError, match='DirectLabelColormap can only'):
        cmap.map('1')


def test_direct_colormap_with_collision():
    # this test assumes that the the selected prime number for hash map size is 11
    color_dict = {
        1: np.array([1, 0, 0, 1]),
        12: np.array([0, 1, 0, 1]),
        23: np.array([0, 0, 1, 1]),
    }
    cmap = DirectLabelColormap(color_dict=color_dict)

    npt.assert_array_equal(cmap.map(1)[0], np.array([1, 0, 0, 1]))
    npt.assert_array_equal(cmap.map(12)[0], np.array([0, 1, 0, 1]))
    npt.assert_array_equal(cmap.map(23)[0], np.array([0, 0, 1, 1]))


def test_direct_colormap_negative_values():
    # Create a DirectLabelColormap with a simple color_dict
    color_dict = {-1: np.array([1, 0, 0, 1]), -2: np.array([0, 1, 0, 1])}
    cmap = DirectLabelColormap(color_dict=color_dict)

    # Map a single value
    mapped = cmap.map(np.int8(-1))
    npt.assert_array_equal(mapped[0], np.array([1, 0, 0, 1]))

    # Map multiple values
    mapped = cmap.map(np.array([-1, -2], dtype=np.int8))
    npt.assert_array_equal(mapped, np.array([[1, 0, 0, 1], [0, 1, 0, 1]]))
