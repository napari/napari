from itertools import cycle

import numpy as np
import pytest
from vispy.color import ColorArray

from napari.layers.utils.color_transformations import (
    normalize_and_broadcast_colors,
    transform_color_cycle,
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
    with pytest.warns(UserWarning):
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
    with pytest.warns(UserWarning):
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
    with pytest.warns(UserWarning):
        colorarray = normalize_and_broadcast_colors(len(data), colors[:-1])
    np.testing.assert_array_equal(colorarray, colors)


def test_normalize_colors_zero_colors():
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    real = np.ones((shape[0], 4), dtype=np.float32)
    with pytest.warns(UserWarning):
        colorarray = normalize_and_broadcast_colors(len(data), [])
    np.testing.assert_array_equal(colorarray, real)


def test_transform_color_cycle():
    colors = ['red', 'blue']
    transformed_color_cycle, transformed_colors = transform_color_cycle(
        colors, elem_name='face_color', default='white'
    )
    transformed_result = np.array(
        [next(transformed_color_cycle) for i in range(10)]
    )

    color_cycle = cycle(np.array([[1, 0, 0, 1], [0, 0, 1, 1]]))
    color_cycle_result = np.array([next(color_cycle) for i in range(10)])

    np.testing.assert_allclose(transformed_result, color_cycle_result)
