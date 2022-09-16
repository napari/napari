import numpy as np
import pytest

from napari._vispy.layers.points import VispyPointsLayer
from napari.layers import Points


@pytest.mark.parametrize("opacity", [0, 0.3, 0.7, 1])
def test_VispyPointsLayer(opacity):
    points = np.array([[100, 100], [200, 200], [300, 100]])
    layer = Points(points, size=30, opacity=opacity)
    visual = VispyPointsLayer(layer)
    assert visual.node.opacity == opacity


def test_remove_selected_with_derived_text():
    """See https://github.com/napari/napari/issues/3504"""
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Points(points, text='class', properties=properties)
    vispy_layer = VispyPointsLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.selected_data = {1}
    layer.remove_selected()

    np.testing.assert_array_equal(text_node.text, ['A', 'C'])


def test_change_text_updates_node_string():
    points = np.random.rand(3, 2)
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'name': np.array(['D', 'E', 'F']),
    }
    layer = Points(points, text='class', properties=properties)
    vispy_layer = VispyPointsLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, properties['class'])

    layer.text = 'name'

    np.testing.assert_array_equal(text_node.text, properties['name'])


def test_change_text_color_updates_node_color():
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    text = {'string': 'class', 'color': [1, 0, 0]}
    layer = Points(points, text=text, properties=properties)
    vispy_layer = VispyPointsLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.color.rgb, [[1, 0, 0]])

    layer.text.color = [0, 0, 1]

    np.testing.assert_array_equal(text_node.color.rgb, [[0, 0, 1]])


def test_change_properties_updates_node_strings():
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Points(points, properties=properties, text='class')
    vispy_layer = VispyPointsLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties = {'class': np.array(['D', 'E', 'F'])}

    np.testing.assert_array_equal(text_node.text, ['D', 'E', 'F'])


def test_update_property_value_then_refresh_text_updates_node_strings():
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Points(points, properties=properties, text='class')
    vispy_layer = VispyPointsLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties['class'][1] = 'D'
    layer.refresh_text()

    np.testing.assert_array_equal(text_node.text, ['A', 'D', 'C'])


def test_change_canvas_size_limits():
    points = np.random.rand(3, 2)
    layer = Points(points, canvas_size_limits=(0, 10000))
    vispy_layer = VispyPointsLayer(layer)
    node = vispy_layer.node

    assert node.canvas_size_limits == (0, 10000)
    layer.canvas_size_limits = (20, 80)
    assert node.canvas_size_limits == (20, 80)


def test_text_with_non_empty_constant_string():
    points = np.random.rand(3, 2)
    layer = Points(points, text={'string': {'constant': 'a'}})

    vispy_layer = VispyPointsLayer(layer)

    text_node = vispy_layer._get_text_node()
    # Vispy cannot broadcast a constant string and assert_array_equal
    # automatically broadcasts, so explicitly check length.
    assert len(text_node.text) == 3
    np.testing.assert_array_equal(text_node.text, ['a', 'a', 'a'])


def test_change_antialiasing():
    """Changing antialiasing on the layer should change it on the vispy node."""
    points = np.random.rand(3, 2)
    layer = Points(points)
    vispy_layer = VispyPointsLayer(layer)
    layer.antialiasing = 5
    assert vispy_layer.node.antialias == layer.antialiasing
