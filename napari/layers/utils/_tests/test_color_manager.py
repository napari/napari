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


color_str = 'red'
color_list = [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]
color_arr = np.asarray(color_list)


@pytest.mark.parametrize('color', [color_str, color_list, color_arr])
def test_set_color_direct(color):
    cm = ColorManager()
    cm.set_color(color=color, n_colors=3, properties={})

    expected_colors = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
    np.testing.assert_allclose(cm.colors, expected_colors)


def test_invalid_color():
    cm = ColorManager()
    with pytest.raises(ValueError):
        cm.set_color(color=42, n_colors=3, properties={})


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
    cm = ColorManager(color_cycle=color_cycle)
    cm.set_color(color='point_type', n_colors=3, properties=properties)
    color_array = transform_color(
        list(islice(cycle(color_cycle), 0, n_colors))
    )
    np.testing.assert_allclose(cm.colors, color_array)

    # # Add new point and test its color
    # coord = [18, 18]
    # layer.selected_data = {0}
    # layer.add(coord)
    # layer_color = getattr(layer, f'{attribute}_color')
    # assert len(layer_color) == shape[0] + 1
    # np.testing.assert_allclose(
    #     layer_color, np.vstack((color_array, transform_color('red'))),
    # )
    #
    # # Check removing data adjusts colors correctly
    # layer.selected_data = {0, 2}
    # layer.remove_selected()
    # assert len(layer.data) == shape[0] - 1
    #
    # layer_color = getattr(layer, f'{attribute}_color')
    # assert len(layer_color) == shape[0] - 1
    # np.testing.assert_allclose(
    #     layer_color,
    #     np.vstack((color_array[1], color_array[3:], transform_color('red'))),
    # )
    #
    # # refresh colors
    # layer.refresh_colors(update_color_mapping=True)
    #
    # # test adding a point with a new property value
    # layer.selected_data = {}
    # current_properties = layer.current_properties
    # current_properties['point_type'] = np.array(['new'])
    # layer.current_properties = current_properties
    # layer.add([10, 10])
    # color_cycle_map = getattr(layer, f'{attribute}_color_cycle_map')
    #
    # assert 'new' in color_cycle_map
    # np.testing.assert_allclose(
    #     color_cycle_map['new'], np.squeeze(transform_color(color_cycle[0]))
    # )
