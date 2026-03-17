import numpy as np
import pytest
from vispy.color import ColorArray

from napari.layers.utils.color_transformations import (
    normalize_and_broadcast_colors,
    transform_color_with_defaults,
)


def test_transform_color_basic():
    """Test inner method with the same name."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    colorarray = transform_color_with_defaults(
        num_entries=len(data),
        colors='r',
        elem_name='edge_color',
        default='black',
    )
    np.testing.assert_array_equal(colorarray, ColorArray('r').rgba)


def test_transform_color_wrong_colorname():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    with pytest.warns(
        UserWarning, match='resetting all edge_color values to black'
    ):
        colorarray = transform_color_with_defaults(
            num_entries=len(data),
            colors='rr',
            elem_name='edge_color',
            default='black',
        )
    np.testing.assert_array_equal(colorarray, ColorArray('black').rgba)


def test_transform_color_wrong_colorlen():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    with pytest.warns(UserWarning, match='Setting face_color to black'):
        colorarray = transform_color_with_defaults(
            num_entries=len(data),
            colors=['r', 'r'],
            elem_name='face_color',
            default='black',
        )
    np.testing.assert_array_equal(colorarray, ColorArray('black').rgba)


def test_normalize_colors_basic():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    colors = ColorArray(['w'] * shape[0]).rgba
    colorarray = normalize_and_broadcast_colors(len(data), colors)
    np.testing.assert_array_equal(colorarray, colors)


def test_normalize_colors_wrong_num():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    colors = ColorArray(['w'] * shape[0]).rgba
    with pytest.warns(
        UserWarning, match='The number of supplied colors mismatch'
    ):
        colorarray = normalize_and_broadcast_colors(len(data), colors[:-1])
    np.testing.assert_array_equal(colorarray, colors)


def test_normalize_colors_zero_colors():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    real = np.ones((shape[0], 4), dtype=np.float32)
    with pytest.warns(
        UserWarning, match='The number of supplied colors mismatch'
    ):
        colorarray = normalize_and_broadcast_colors(len(data), [])
    np.testing.assert_array_equal(colorarray, real)
