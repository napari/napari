import json
from itertools import cycle

import numpy as np
import pytest

from napari.utils.colormaps.categorical_colormap import CategoricalColormap


def test_default_categorical_colormap():
    cmap = CategoricalColormap()
    assert cmap.colormap == {}

    color_cycle = cmap.fallback_color
    np.testing.assert_almost_equal(color_cycle.values, [[1, 1, 1, 1]])
    np.testing.assert_almost_equal(next(color_cycle.cycle), [1, 1, 1, 1])


def test_categorical_colormap_direct():
    """Test a categorical colormap with a provided mapping"""
    colormap = {'hi': np.array([1, 1, 1, 1]), 'hello': np.array([0, 0, 0, 0])}
    cmap = CategoricalColormap(colormap=colormap)

    color = cmap.map(['hi'])
    np.testing.assert_allclose(color, [[1, 1, 1, 1]])
    color = cmap.map(['hello'])
    np.testing.assert_allclose(color, [[0, 0, 0, 0]])

    # test that the default fallback color (white) is applied
    new_color_0 = cmap.map(['not a key'])
    np.testing.assert_almost_equal(new_color_0, [[1, 1, 1, 1]])
    new_cmap = cmap.colormap
    np.testing.assert_almost_equal(new_cmap['not a key'], [1, 1, 1, 1])

    # set a cycle of fallback colors
    new_fallback_colors = [[1, 0, 0, 1], [0, 1, 0, 1]]
    cmap.fallback_color = new_fallback_colors
    new_color_1 = cmap.map(['new_prop 1'])
    np.testing.assert_almost_equal(
        np.squeeze(new_color_1), new_fallback_colors[0]
    )
    new_color_2 = cmap.map(['new_prop 2'])
    np.testing.assert_almost_equal(
        np.squeeze(new_color_2), new_fallback_colors[1]
    )


def test_categorical_colormap_cycle():
    color_cycle = [[1, 1, 1, 1], [1, 0, 0, 1]]
    cmap = CategoricalColormap(fallback_color=color_cycle)

    # verify that no mapping between prop value and color has been set
    assert cmap.colormap == {}

    # the values used to create the color cycle can be accessed via fallback color
    np.testing.assert_almost_equal(cmap.fallback_color.values, color_cycle)

    # map 2 colors, verify their colors are returned in order
    colors = cmap.map(['hi', 'hello'])
    np.testing.assert_almost_equal(colors, color_cycle)

    # map a third color and verify the colors wrap around
    third_color = cmap.map(['bonjour'])
    np.testing.assert_almost_equal(np.squeeze(third_color), color_cycle[0])


def test_categorical_colormap_cycle_as_dict():
    color_values = np.array([[1, 1, 1, 1], [1, 0, 0, 1]])
    color_cycle = cycle(color_values)
    fallback_color = {'values': color_values, 'cycle': color_cycle}
    cmap = CategoricalColormap(fallback_color=fallback_color)

    # verify that no mapping between prop value and color has been set
    assert cmap.colormap == {}

    # the values used to create the color cycle can be accessed via fallback color
    np.testing.assert_almost_equal(cmap.fallback_color.values, color_values)
    np.testing.assert_almost_equal(
        next(cmap.fallback_color.cycle), color_values[0]
    )


fallback_colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])


def test_categorical_colormap_from_array():

    cmap = CategoricalColormap.from_array(fallback_colors)
    np.testing.assert_almost_equal(cmap.fallback_color.values, fallback_colors)


color_mapping = {
    'typeA': np.array([1, 1, 1, 1]),
    'typeB': np.array([1, 0, 0, 1]),
}
default_fallback_color = np.array([[1, 1, 1, 1]])


@pytest.mark.parametrize(
    'params,expected',
    [
        ({'colormap': color_mapping}, (color_mapping, default_fallback_color)),
        (
            {'colormap': color_mapping, 'fallback_color': fallback_colors},
            (color_mapping, fallback_colors),
        ),
        ({'fallback_color': fallback_colors}, ({}, fallback_colors)),
        (color_mapping, (color_mapping, default_fallback_color)),
    ],
)
def test_categorical_colormap_from_dict(params, expected):
    cmap = CategoricalColormap.from_dict(params)
    np.testing.assert_equal(cmap.colormap, expected[0])
    np.testing.assert_almost_equal(cmap.fallback_color.values, expected[1])


def test_categorical_colormap_equality():
    color_cycle = [[1, 1, 1, 1], [1, 0, 0, 1]]
    cmap_1 = CategoricalColormap(fallback_color=color_cycle)
    cmap_2 = CategoricalColormap(fallback_color=color_cycle)
    cmap_3 = CategoricalColormap(fallback_color=[[1, 1, 1, 1], [1, 1, 0, 1]])
    cmap_4 = CategoricalColormap(
        colormap={0: np.array([0, 0, 0, 1])}, fallback_color=color_cycle
    )
    assert cmap_1 == cmap_2
    assert cmap_1 != cmap_3
    assert cmap_1 != cmap_4

    # test equality against a different type
    assert cmap_1 != color_cycle


@pytest.mark.parametrize(
    'params',
    [
        {'colormap': color_mapping},
        {'colormap': color_mapping, 'fallback_color': fallback_colors},
        {'fallback_color': fallback_colors},
    ],
)
def test_categorical_colormap_serialization(params):
    cmap_1 = CategoricalColormap(**params)
    cmap_json = cmap_1.json()

    json_dict = json.loads(cmap_json)
    cmap_2 = CategoricalColormap(**json_dict)
    assert cmap_1 == cmap_2
