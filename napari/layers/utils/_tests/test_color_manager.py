from itertools import cycle, islice

import numpy as np
import pytest

from napari.layers.utils.color_manager import ColorManager
from napari.utils.colormaps.standardize_color import transform_color


def _make_cycled_properties(values, length):
    """Helper function to make property values

    Parameters
    ----------
    values
        The values to be cycled.
    length : int
        The length of the resulting property array

    Returns
    -------
    cycled_properties : np.ndarray
        The property array comprising the cycled values.
    """
    cycled_properties = np.array(list(islice(cycle(values), 0, length)))
    return cycled_properties


def test_color_manager_empty():
    cm = ColorManager()
    np.testing.assert_allclose(cm.values, np.empty((0, 4)))


color_str = 'red'
color_list = [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]
color_arr = np.asarray(color_list)


@pytest.mark.parametrize('color', [color_str, color_list, color_arr])
def test_set_color_direct(color):
    cm = ColorManager(colors=color, n_colors=3, properties={})

    expected_colors = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
    np.testing.assert_allclose(cm.values, expected_colors)


def test_invalid_color():
    with pytest.raises(TypeError):
        ColorManager(colors=42, n_colors=3, properties={})


color_cycle_str = ['red', 'blue']
color_cycle_rgb = [[1, 0, 0], [0, 0, 1]]
color_cycle_rgba = [[1, 0, 0, 1], [0, 0, 1, 1]]


@pytest.mark.parametrize(
    "color_cycle", [color_cycle_str, color_cycle_rgb, color_cycle_rgba],
)
def test_color_cycle(color_cycle):
    """Test setting color with a color cycle list"""
    # create Points using list color cycle
    n_colors = 10
    properties = {'point_type': _make_cycled_properties(['A', 'B'], n_colors)}
    cm = ColorManager(
        colors='point_type',
        n_colors=n_colors,
        properties=properties,
        categorical_colormap=color_cycle,
    )
    color_array = transform_color(
        list(islice(cycle(color_cycle), 0, n_colors))
    )
    np.testing.assert_allclose(cm.values, color_array)

    # Add 2 color elements and test their color
    cm.add(['A'], n_colors=2)
    cm_colors = cm.values
    assert len(cm_colors) == n_colors + 2
    np.testing.assert_allclose(
        cm_colors,
        np.vstack(
            (color_array, transform_color('red'), transform_color('red'))
        ),
    )

    # Check removing data adjusts colors correctly
    cm.remove(set([0, 2, 11]))
    cm_colors_2 = cm.values
    assert len(cm_colors_2) == (n_colors - 1)
    np.testing.assert_allclose(
        cm_colors_2,
        np.vstack((color_array[1], color_array[3:], transform_color('red'))),
    )
