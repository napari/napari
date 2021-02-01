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
    np.testing.assert_allclose(cm.colors, np.empty((0, 4)))
    assert cm.mode == 'direct'


color_str = 'red'
color_list = [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]
color_arr = np.asarray(color_list)


@pytest.mark.parametrize('color', [color_str, color_list, color_arr])
def test_set_color_direct(color):
    cm = ColorManager(colors=color, n_colors=3, mode='direct')
    color_mode = cm.mode
    assert color_mode == 'direct'
    expected_colors = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
    np.testing.assert_allclose(cm.colors, expected_colors)
    np.testing.assert_allclose(cm.current_color, expected_colors[-1])


def test_continuous_colormap():
    # create ColorManager with a continuous colormap
    n_colors = 10
    properties = {'point_type': _make_cycled_properties([0, 1.5], n_colors)}
    cm = ColorManager(
        n_colors=n_colors,
        color_properties=properties,
        continuous_colormap='gray',
        mode='colormap',
    )
    color_mode = cm.mode
    assert color_mode == 'colormap'
    color_array = transform_color(['black', 'white'] * int(n_colors / 2))
    colors = cm.colors.copy()
    np.testing.assert_allclose(colors, color_array)
    assert cm.current_color == properties['point_type'][-1]

    # Add 2 color elements and test their color
    cm.add(0, n_colors=2)
    cm_colors = cm.colors
    assert len(cm_colors) == n_colors + 2
    np.testing.assert_allclose(
        cm_colors,
        np.vstack(
            (color_array, transform_color('black'), transform_color('black'))
        ),
    )

    # Check removing data adjusts colors correctly
    cm.remove({0, 2, 11})
    cm_colors_2 = cm.colors
    assert len(cm_colors_2) == (n_colors - 1)
    np.testing.assert_allclose(
        cm_colors_2,
        np.vstack((color_array[1], color_array[3:], transform_color('black'))),
    )

    # adjust the clims
    cm.contrast_limits = (0, 3)
    updated_colors = cm.colors
    np.testing.assert_allclose(updated_colors[-2], [0.5, 0.5, 0.5, 1])

    # first verify that prop value 0 is colored black
    current_colors = cm.colors
    np.testing.assert_allclose(current_colors[-1], [0, 0, 0, 1])

    # change the colormap
    new_colormap = 'gray_r'
    cm.continuous_colormap = new_colormap
    assert cm.continuous_colormap.name == new_colormap

    # the props valued 0 should now be white
    updated_colors = cm.colors
    np.testing.assert_allclose(updated_colors[-1], [1, 1, 1, 1])
