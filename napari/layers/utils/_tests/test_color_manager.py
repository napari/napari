import json
from itertools import cycle, islice

import numpy as np
import pytest
from pydantic import ValidationError

from napari.layers.utils.color_manager import ColorManager, ColorProperties
from napari.utils.colormaps.categorical_colormap import CategoricalColormap
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
    assert cm.color_mode == 'direct'


color_mapping = {0: np.array([1, 1, 1, 1]), 1: np.array([1, 0, 0, 1])}
fallback_colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])
default_fallback_color = np.array([[1, 1, 1, 1]])
categorical_map = CategoricalColormap(
    colormap=color_mapping, fallback_color=fallback_colors
)


@pytest.mark.parametrize(
    'cat_cmap,expected',
    [
        ({'colormap': color_mapping}, (color_mapping, default_fallback_color)),
        (
            {'colormap': color_mapping, 'fallback_color': fallback_colors},
            (color_mapping, fallback_colors),
        ),
        ({'fallback_color': fallback_colors}, ({}, fallback_colors)),
        (color_mapping, (color_mapping, default_fallback_color)),
        (fallback_colors, ({}, fallback_colors)),
        (categorical_map, (color_mapping, fallback_colors)),
    ],
)
def test_categorical_colormap_from_dict(cat_cmap, expected):
    colors = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [0, 0, 0, 1]])
    cm = ColorManager(
        colors=colors, categorical_colormap=cat_cmap, color_mode='direct'
    )
    np.testing.assert_equal(cm.categorical_colormap.colormap, expected[0])
    np.testing.assert_almost_equal(
        cm.categorical_colormap.fallback_color.values, expected[1]
    )


def test_invalid_categorical_colormap():
    colors = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [0, 0, 0, 1]])
    invalid_cmap = 42
    with pytest.raises(ValidationError):
        _ = ColorManager(
            colors=colors,
            categorical_colormap=invalid_cmap,
            color_mode='direct',
        )


c_prop_dict = {
    'name': 'point_type',
    'values': np.array(['A', 'B', 'C']),
    'current_value': np.array(['C']),
}
c_prop_obj = ColorProperties(**c_prop_dict)


@pytest.mark.parametrize(
    'c_props,expected',
    [
        (None, None),
        ({}, None),
        (c_prop_obj, c_prop_obj),
        (c_prop_dict, c_prop_obj),
    ],
)
def test_color_properties_coercion(c_props, expected):
    colors = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [0, 0, 0, 1]])
    cm = ColorManager(
        colors=colors, color_properties=c_props, color_mode='direct'
    )
    assert cm.color_properties == expected


wrong_type = ('prop_1', np.array([1, 2, 3]))
invalid_keys = {'values': np.array(['A', 'B', 'C'])}


@pytest.mark.parametrize('c_props', [wrong_type, invalid_keys])
def test_invalid_color_properties(c_props):
    colors = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [0, 0, 0, 1]])
    with pytest.raises(ValidationError):
        _ = ColorManager(
            colors=colors, color_properties=c_props, color_mode='direct'
        )


@pytest.mark.parametrize(
    'curr_color,expected',
    [
        (None, np.array([0, 0, 0, 1])),
        ([], np.array([0, 0, 0, 1])),
        ('red', np.array([1, 0, 0, 1])),
        ([1, 0, 0, 1], np.array([1, 0, 0, 1])),
    ],
)
def test_current_color_coercion(curr_color, expected):
    colors = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [0, 0, 0, 1]])
    cm = ColorManager(
        colors=colors, current_color=curr_color, color_mode='direct'
    )
    np.testing.assert_allclose(cm.current_color, expected)


color_str = ['red', 'red', 'red']
color_list = [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]
color_arr = np.asarray(color_list)


@pytest.mark.parametrize('color', [color_str, color_list, color_arr])
def test_color_manager_direct(color):
    cm = ColorManager(colors=color, color_mode='direct')
    color_mode = cm.color_mode
    assert color_mode == 'direct'
    expected_colors = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
    np.testing.assert_allclose(cm.colors, expected_colors)
    np.testing.assert_allclose(cm.current_color, expected_colors[-1])

    # test adding a color
    new_color = [1, 1, 1, 1]
    cm._add(new_color)
    np.testing.assert_allclose(cm.colors[-1], new_color)

    # test removing colors
    cm._remove([0, 3])
    np.testing.assert_allclose(cm.colors, expected_colors[1:3])

    # test pasting colors
    paste_colors = np.array([[0, 0, 0, 1], [0, 0, 0, 1]])
    cm._paste(colors=paste_colors, properties={})
    post_paste_colors = np.vstack((expected_colors[1:3], paste_colors))
    np.testing.assert_allclose(cm.colors, post_paste_colors)

    # refreshing the colors in direct mode should have no effect
    cm._refresh_colors(properties={})
    np.testing.assert_allclose(cm.colors, post_paste_colors)


@pytest.mark.parametrize('color', [color_str, color_list, color_arr])
def test_set_color_direct(color):
    """Test setting the colors via the set_color method in direct mode"""
    # create an empty color manager
    cm = ColorManager()
    np.testing.assert_allclose(cm.colors, np.empty((0, 4)))
    assert cm.color_mode == 'direct'

    # set colors
    expected_colors = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
    cm._set_color(
        color, n_colors=len(color), properties={}, current_properties={}
    )
    np.testing.assert_almost_equal(cm.colors, expected_colors)


def test_continuous_colormap():
    # create ColorManager with a continuous colormap
    n_colors = 10
    properties = {
        'name': 'point_type',
        'values': _make_cycled_properties([0, 1.5], n_colors),
    }
    cm = ColorManager(
        color_properties=properties,
        continuous_colormap='gray',
        color_mode='colormap',
    )
    color_mode = cm.color_mode
    assert color_mode == 'colormap'
    color_array = transform_color(['black', 'white'] * int(n_colors / 2))
    colors = cm.colors.copy()
    np.testing.assert_allclose(colors, color_array)
    np.testing.assert_allclose(cm.current_color, [1, 1, 1, 1])

    # Add 2 color elements and test their color
    cm._add(0, n_colors=2)
    cm_colors = cm.colors
    assert len(cm_colors) == n_colors + 2
    np.testing.assert_allclose(
        cm_colors,
        np.vstack(
            (color_array, transform_color('black'), transform_color('black'))
        ),
    )

    # Check removing data adjusts colors correctly
    cm._remove({0, 2, 11})
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

    # test pasting values
    paste_props = {'point_type': np.array([0, 0])}
    paste_colors = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    cm._paste(colors=paste_colors, properties=paste_props)
    np.testing.assert_allclose(cm.colors[-2:], paste_colors)


def test_set_color_colormap():
    # make an empty colormanager
    init_color_properties = {
        'name': 'point_type',
        'values': np.empty(0),
        'current_value': np.array([1.5]),
    }
    cm = ColorManager(
        color_properties=init_color_properties,
        continuous_colormap='gray',
        color_mode='colormap',
    )

    # use the set_color method to update the colors
    n_colors = 10
    updated_properties = {
        'point_type': _make_cycled_properties([0, 1.5], n_colors)
    }
    current_properties = {'point_type': np.array([1.5])}
    cm._set_color(
        color='point_type',
        n_colors=n_colors,
        properties=updated_properties,
        current_properties=current_properties,
    )
    color_array = transform_color(['black', 'white'] * int(n_colors / 2))
    np.testing.assert_allclose(cm.colors, color_array)


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
        color_mode='cycle',
        color_properties=properties,
        categorical_colormap=color_cycle,
    )
    color_mode = cm.color_mode
    assert color_mode == 'cycle'
    color_array = transform_color(
        list(islice(cycle(color_cycle), 0, n_colors))
    )
    np.testing.assert_allclose(cm.colors, color_array)

    # Add 2 color elements and test their color
    cm._add('A', n_colors=2)
    cm_colors = cm.colors
    assert len(cm_colors) == n_colors + 2
    np.testing.assert_allclose(
        cm_colors,
        np.vstack(
            (color_array, transform_color('red'), transform_color('red'))
        ),
    )

    # Check removing data adjusts colors correctly
    cm._remove({0, 2, 11})
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

    # test pasting values
    paste_props = {'point_type': np.array(['B', 'B'])}
    paste_colors = np.array([[0, 0, 0, 1], [0, 0, 0, 1]])
    cm._paste(colors=paste_colors, properties=paste_props)
    np.testing.assert_allclose(cm.colors[-2:], paste_colors)


def test_set_color_cycle():
    # make an empty colormanager
    init_color_properties = {
        'name': 'point_type',
        'values': np.empty(0),
        'current_value': np.array(['A']),
    }
    cm = ColorManager(
        color_properties=init_color_properties,
        categorical_colormap=['black', 'white'],
        mode='cycle',
    )

    # use the set_color method to update the colors
    n_colors = 10
    updated_properties = {
        'point_type': _make_cycled_properties(['A', 'B'], n_colors)
    }
    current_properties = {'point_type': np.array(['B'])}
    cm._set_color(
        color='point_type',
        n_colors=n_colors,
        properties=updated_properties,
        current_properties=current_properties,
    )
    color_array = transform_color(['black', 'white'] * int(n_colors / 2))
    np.testing.assert_allclose(cm.colors, color_array)


@pytest.mark.parametrize('n_colors', [0, 1, 5])
def test_init_color_manager_direct(n_colors):
    color_manager = ColorManager._from_layer_kwargs(
        colors='red',
        properties={},
        n_colors=n_colors,
        continuous_colormap='viridis',
        contrast_limits=None,
        categorical_colormap=[[0, 0, 0, 1], [1, 1, 1, 1]],
    )

    assert len(color_manager.colors) == n_colors
    assert color_manager.color_mode == 'direct'
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
    color_manager_2 = ColorManager._from_layer_kwargs(
        colors=cm_dict, properties={}, n_colors=n_colors
    )
    assert color_manager == color_manager_2

    # test json serialization
    json_str = color_manager.json()
    cm_json_dict = json.loads(json_str)
    color_manager_3 = ColorManager._from_layer_kwargs(
        colors=cm_json_dict, properties={}, n_colors=n_colors
    )
    assert color_manager == color_manager_3


def test_init_color_manager_cycle():
    n_colors = 10
    color_cycle = [[0, 0, 0, 1], [1, 1, 1, 1]]
    properties = {'point_type': _make_cycled_properties(['A', 'B'], n_colors)}
    color_manager = ColorManager._from_layer_kwargs(
        colors='point_type',
        properties=properties,
        n_colors=n_colors,
        continuous_colormap='viridis',
        contrast_limits=None,
        categorical_colormap=color_cycle,
    )

    assert len(color_manager.colors) == n_colors
    assert color_manager.color_mode == 'cycle'
    color_array = transform_color(
        list(islice(cycle(color_cycle), 0, n_colors))
    )
    np.testing.assert_allclose(color_manager.colors, color_array)
    assert color_manager.color_properties.current_value == 'B'

    # test that colormanager state can be saved and loaded
    cm_dict = color_manager.dict()
    color_manager_2 = ColorManager._from_layer_kwargs(
        colors=cm_dict, properties=properties
    )
    assert color_manager == color_manager_2

    # test json serialization
    json_str = color_manager.json()
    cm_json_dict = json.loads(json_str)
    color_manager_3 = ColorManager._from_layer_kwargs(
        colors=cm_json_dict, properties={}, n_colors=n_colors
    )
    assert color_manager == color_manager_3


def test_init_color_manager_cycle_with_colors_dict():
    """Test initializing color cycle ColorManager from layer kwargs
    where the colors are given as a dictionary of ColorManager
    fields/values
    """
    n_colors = 10
    color_cycle = [[0, 0, 0, 1], [1, 1, 1, 1]]
    properties = {'point_type': _make_cycled_properties(['A', 'B'], n_colors)}
    colors_dict = {
        'color_properties': 'point_type',
        'color_mode': 'cycle',
        'categorical_colormap': color_cycle,
    }
    color_manager = ColorManager._from_layer_kwargs(
        colors=colors_dict,
        properties=properties,
        n_colors=n_colors,
        continuous_colormap='viridis',
    )
    assert len(color_manager.colors) == n_colors
    assert color_manager.color_mode == 'cycle'
    color_array = transform_color(
        list(islice(cycle(color_cycle), 0, n_colors))
    )
    np.testing.assert_allclose(color_manager.colors, color_array)
    assert color_manager.color_properties.current_value == 'B'
    assert color_manager.continuous_colormap.name == 'viridis'


def test_init_empty_color_manager_cycle():
    n_colors = 0
    color_cycle = [[0, 0, 0, 1], [1, 1, 1, 1]]
    properties = {'point_type': ['A', 'B']}
    color_manager = ColorManager._from_layer_kwargs(
        colors='point_type',
        properties=properties,
        n_colors=n_colors,
        continuous_colormap='viridis',
        contrast_limits=None,
        categorical_colormap=color_cycle,
    )

    assert len(color_manager.colors) == n_colors
    assert color_manager.color_mode == 'cycle'

    np.testing.assert_allclose(color_manager.current_color, [0, 0, 0, 1])
    assert color_manager.color_properties.current_value == 'A'

    color_manager._add()
    np.testing.assert_allclose(color_manager.colors, [[0, 0, 0, 1]])

    color_manager.color_properties.current_value = 'B'
    color_manager._add()
    np.testing.assert_allclose(
        color_manager.colors, [[0, 0, 0, 1], [1, 1, 1, 1]]
    )

    # test that colormanager state can be saved and loaded
    cm_dict = color_manager.dict()
    color_manager_2 = ColorManager._from_layer_kwargs(
        colors=cm_dict, properties=properties
    )
    assert color_manager == color_manager_2


def test_init_color_manager_colormap():
    n_colors = 10
    color_cycle = [[0, 0, 0, 1], [1, 1, 1, 1]]
    properties = {'point_type': _make_cycled_properties([0, 1.5], n_colors)}
    color_manager = ColorManager._from_layer_kwargs(
        colors='point_type',
        properties=properties,
        n_colors=n_colors,
        continuous_colormap='gray',
        contrast_limits=None,
        categorical_colormap=color_cycle,
    )

    assert len(color_manager.colors) == n_colors
    assert color_manager.color_mode == 'colormap'
    color_array = transform_color(['black', 'white'] * int(n_colors / 2))
    colors = color_manager.colors.copy()
    np.testing.assert_allclose(colors, color_array)
    np.testing.assert_allclose(color_manager.current_color, [1, 1, 1, 1])
    assert color_manager.color_properties.current_value == 1.5

    # test that colormanager state can be saved and loaded
    cm_dict = color_manager.dict()
    color_manager_2 = ColorManager._from_layer_kwargs(
        colors=cm_dict, properties=properties
    )
    assert color_manager == color_manager_2

    # test json serialization
    json_str = color_manager.json()
    cm_json_dict = json.loads(json_str)
    color_manager_3 = ColorManager._from_layer_kwargs(
        colors=cm_json_dict, properties={}, n_colors=n_colors
    )
    assert color_manager == color_manager_3


def test_init_color_manager_colormap_with_colors_dict():
    """Test initializing colormap ColorManager from layer kwargs
    where the colors are given as a dictionary of ColorManager
    fields/values
    """
    n_colors = 10
    color_cycle = [[0, 0, 0, 1], [1, 1, 1, 1]]
    properties = {'point_type': _make_cycled_properties([0, 1.5], n_colors)}
    colors_dict = {
        'color_properties': 'point_type',
        'color_mode': 'colormap',
        'categorical_colormap': color_cycle,
        'continuous_colormap': 'gray',
    }
    color_manager = ColorManager._from_layer_kwargs(
        colors=colors_dict, properties=properties, n_colors=n_colors
    )
    assert len(color_manager.colors) == n_colors
    assert color_manager.color_mode == 'colormap'
    color_array = transform_color(['black', 'white'] * int(n_colors / 2))
    colors = color_manager.colors.copy()
    np.testing.assert_allclose(colors, color_array)
    np.testing.assert_allclose(color_manager.current_color, [1, 1, 1, 1])
    assert color_manager.color_properties.current_value == 1.5
    assert color_manager.continuous_colormap.name == 'gray'


def test_init_empty_color_manager_colormap():
    n_colors = 0
    color_cycle = [[0, 0, 0, 1], [1, 1, 1, 1]]
    properties = {'point_type': [0]}
    color_manager = ColorManager._from_layer_kwargs(
        colors='point_type',
        properties=properties,
        n_colors=n_colors,
        color_mode='colormap',
        continuous_colormap='gray',
        contrast_limits=None,
        categorical_colormap=color_cycle,
    )

    assert len(color_manager.colors) == n_colors
    assert color_manager.color_mode == 'colormap'

    np.testing.assert_allclose(color_manager.current_color, [0, 0, 0, 1])
    assert color_manager.color_properties.current_value == 0

    color_manager._add()
    np.testing.assert_allclose(color_manager.colors, [[1, 1, 1, 1]])

    color_manager.color_properties.current_value = 1.5
    color_manager._add(update_clims=True)
    np.testing.assert_allclose(
        color_manager.colors, [[0, 0, 0, 1], [1, 1, 1, 1]]
    )

    # test that colormanager state can be saved and loaded
    cm_dict = color_manager.dict()
    color_manager_2 = ColorManager._from_layer_kwargs(
        colors=cm_dict, properties=properties
    )
    assert color_manager == color_manager_2


def test_color_manager_invalid_color_properties():
    """Passing an invalid property name for color_properties
    should raise a KeyError
    """
    n_colors = 10
    color_cycle = [[0, 0, 0, 1], [1, 1, 1, 1]]
    properties = {'point_type': _make_cycled_properties([0, 1.5], n_colors)}
    colors_dict = {
        'color_properties': 'not_point_type',
        'color_mode': 'colormap',
        'categorical_colormap': color_cycle,
        'continuous_colormap': 'gray',
    }
    with pytest.raises(KeyError):
        _ = ColorManager._from_layer_kwargs(
            colors=colors_dict, properties=properties, n_colors=n_colors
        )


def test_refresh_colors():
    # create ColorManager with a continuous colormap
    n_colors = 4
    properties = {
        'name': 'point_type',
        'values': _make_cycled_properties([0, 1.5], n_colors),
    }
    cm = ColorManager(
        color_properties=properties,
        continuous_colormap='gray',
        color_mode='colormap',
    )
    color_mode = cm.color_mode
    assert color_mode == 'colormap'
    color_array = transform_color(['black', 'white'] * int(n_colors / 2))
    colors = cm.colors.copy()
    np.testing.assert_allclose(colors, color_array)
    np.testing.assert_allclose(cm.current_color, [1, 1, 1, 1])

    # after refresh, the color should now be white. since we didn't
    # update the color mapping, the other values should remain
    # unchanged even though we added a value that extends the range
    # of values
    new_properties = {'point_type': properties['values']}
    new_properties['point_type'][0] = 3
    cm._refresh_colors(new_properties, update_color_mapping=False)
    new_colors = color_array.copy()
    new_colors[0] = [1, 1, 1, 1]
    np.testing.assert_allclose(cm.colors, new_colors)

    # now, refresh the colors, but update the mapping
    cm._refresh_colors(new_properties, update_color_mapping=True)
    refreshed_colors = [
        [1, 1, 1, 1],
        [0.5, 0.5, 0.5, 1],
        [0, 0, 0, 1],
        [0.5, 0.5, 0.5, 1],
    ]
    np.testing.assert_allclose(cm.colors, refreshed_colors)
