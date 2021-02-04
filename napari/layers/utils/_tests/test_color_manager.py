from itertools import cycle, islice

import numpy as np
import pytest

from napari.layers.utils.color_manager import (
    ColorManager,
    initialize_color_manager,
)
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
    properties = {
        'name': 'point_type',
        'values': _make_cycled_properties([0, 1.5], n_colors),
    }
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
    np.testing.assert_allclose(cm.current_color, [1, 1, 1, 1])

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


color_cycle_str = ['red', 'blue']
color_cycle_rgb = [[1, 0, 0], [0, 0, 1]]
color_cycle_rgba = [[1, 0, 0, 1], [0, 0, 1, 1]]


@pytest.mark.parametrize(
    "color_cycle",
    [color_cycle_str, color_cycle_rgb, color_cycle_rgba],
)
def test_color_cycle(color_cycle):
    """Test setting color with a color cycle list"""
    # create Points using list color cycle
    n_colors = 10
    properties = {
        'name': 'point_type',
        'values': _make_cycled_properties(['A', 'B'], n_colors),
    }
    cm = ColorManager(
        mode='cycle',
        n_colors=n_colors,
        color_properties=properties,
        categorical_colormap=color_cycle,
    )
    color_mode = cm.mode
    assert color_mode == 'cycle'
    color_array = transform_color(
        list(islice(cycle(color_cycle), 0, n_colors))
    )
    np.testing.assert_allclose(cm.colors, color_array)

    # Add 2 color elements and test their color
    cm.add('A', n_colors=2)
    cm_colors = cm.colors
    assert len(cm_colors) == n_colors + 2
    np.testing.assert_allclose(
        cm_colors,
        np.vstack(
            (color_array, transform_color('red'), transform_color('red'))
        ),
    )

    # Check removing data adjusts colors correctly
    cm.remove({0, 2, 11})
    cm_colors_2 = cm.colors
    assert len(cm_colors_2) == (n_colors - 1)
    np.testing.assert_allclose(
        cm_colors_2,
        np.vstack((color_array[1], color_array[3:], transform_color('red'))),
    )

    # update the colormap
    cm.categorical_colormap = ['black', 'white']

    # the first color should now be black
    np.testing.assert_allclose(cm.colors[0], [0, 0, 0, 1])


@pytest.mark.parametrize('n_colors', [0, 1, 5])
def test_init_color_manager_direct(n_colors):
    color_manager = initialize_color_manager(
        n_colors=n_colors,
        colors='red',
        mode='direct',
        continuous_colormap='viridis',
        contrast_limits=None,
        categorical_colormap=[[0, 0, 0, 1], [1, 1, 1, 1]],
        properties={},
    )

    assert len(color_manager.colors) == n_colors
    assert color_manager.mode == 'direct'
    np.testing.assert_array_almost_equal(
        color_manager.current_color, [1, 0, 0, 1]
    )
    if n_colors > 0:
        expected_colors = np.tile([1, 0, 0, 1], (n_colors, 1))
        np.testing.assert_array_almost_equal(
            color_manager.colors, expected_colors
        )
    # test that colormanager state can be saved and loaded
    cm_dict = color_manager.dict()
    color_manager_2 = initialize_color_manager(colors=cm_dict, properties={})
    assert color_manager == color_manager_2


def test_init_color_manager_cycle():
    n_colors = 10
    color_cycle = [[0, 0, 0, 1], [1, 1, 1, 1]]
    properties = {'point_type': _make_cycled_properties(['A', 'B'], n_colors)}
    color_manager = initialize_color_manager(
        n_colors=n_colors,
        colors='point_type',
        mode='cycle',
        continuous_colormap='viridis',
        contrast_limits=None,
        categorical_colormap=color_cycle,
        properties=properties,
    )

    assert len(color_manager.colors) == n_colors
    assert color_manager.mode == 'cycle'
    color_array = transform_color(
        list(islice(cycle(color_cycle), 0, n_colors))
    )
    np.testing.assert_allclose(color_manager.colors, color_array)
    assert color_manager.color_properties.current_value == 'B'

    # test that colormanager state can be saved and loaded
    cm_dict = color_manager.dict()
    color_manager_2 = initialize_color_manager(
        colors=cm_dict, properties=properties
    )
    assert color_manager == color_manager_2


def test_init_color_manager_colormap():
    n_colors = 10
    color_cycle = [[0, 0, 0, 1], [1, 1, 1, 1]]
    properties = {'point_type': _make_cycled_properties([0, 1.5], n_colors)}
    color_manager = initialize_color_manager(
        n_colors=n_colors,
        colors='point_type',
        mode='colormap',
        continuous_colormap='gray',
        contrast_limits=None,
        categorical_colormap=color_cycle,
        properties=properties,
    )

    assert len(color_manager.colors) == n_colors
    assert color_manager.mode == 'colormap'
    color_array = transform_color(['black', 'white'] * int(n_colors / 2))
    colors = color_manager.colors.copy()
    np.testing.assert_allclose(colors, color_array)
    np.testing.assert_allclose(color_manager.current_color, [1, 1, 1, 1])
    assert color_manager.color_properties.current_value == 1.5

    # test that colormanager state can be saved and loaded
    cm_dict = color_manager.dict()
    color_manager_2 = initialize_color_manager(
        colors=cm_dict, properties=properties
    )
    assert color_manager == color_manager_2
