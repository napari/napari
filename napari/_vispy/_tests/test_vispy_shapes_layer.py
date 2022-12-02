import numpy as np

from napari._vispy.layers.shapes import VispyShapesLayer
from napari.layers import Shapes


def test_remove_selected_with_derived_text():
    """See https://github.com/napari/napari/issues/3504"""
    np.random.seed(0)
    shapes = np.random.rand(3, 4, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Shapes(shapes, properties=properties, text='class')
    vispy_layer = VispyShapesLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.selected_data = {1}
    layer.remove_selected()

    np.testing.assert_array_equal(text_node.text, ['A', 'C'])


def test_change_text_updates_node_string():
    np.random.seed(0)
    shapes = np.random.rand(3, 4, 2)
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'name': np.array(['D', 'E', 'F']),
    }
    layer = Shapes(shapes, properties=properties, text='class')
    vispy_layer = VispyShapesLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, properties['class'])

    layer.text = 'name'

    np.testing.assert_array_equal(text_node.text, properties['name'])


def test_change_text_color_updates_node_color():
    np.random.seed(0)
    shapes = np.random.rand(3, 4, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    text = {'string': 'class', 'color': [1, 0, 0]}
    layer = Shapes(shapes, properties=properties, text=text)
    vispy_layer = VispyShapesLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.color.rgb, [[1, 0, 0]])

    layer.text.color = [0, 0, 1]

    np.testing.assert_array_equal(text_node.color.rgb, [[0, 0, 1]])


def test_change_properties_updates_node_strings():
    np.random.seed(0)
    shapes = np.random.rand(3, 4, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Shapes(shapes, properties=properties, text='class')
    vispy_layer = VispyShapesLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties = {'class': np.array(['D', 'E', 'F'])}

    np.testing.assert_array_equal(text_node.text, ['D', 'E', 'F'])


def test_update_property_value_then_refresh_text_updates_node_strings():
    np.random.seed(0)
    shapes = np.random.rand(3, 4, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Shapes(shapes, properties=properties, text='class')
    vispy_layer = VispyShapesLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties['class'][1] = 'D'
    layer.refresh_text()

    np.testing.assert_array_equal(text_node.text, ['A', 'D', 'C'])


def test_text_with_non_empty_constant_string():
    np.random.seed(0)
    shapes = np.random.rand(3, 4, 2)
    layer = Shapes(shapes, text={'string': {'constant': 'a'}})

    vispy_layer = VispyShapesLayer(layer)

    text_node = vispy_layer._get_text_node()
    # Vispy cannot broadcast a constant string and assert_array_equal
    # automatically broadcasts, so explicitly check length.
    assert len(text_node.text) == 3
    np.testing.assert_array_equal(text_node.text, ['a', 'a', 'a'])

    # Ensure we do position calculation for constants.
    # See https://github.com/napari/napari/issues/5378
    expected_position = np.mean(shapes, axis=1)
    # Skip 3rd dimension of vispy position for 2D data, and also flip
    # xy coordinates to be rc.
    actual_position = text_node.pos[:, 1::-1]
    np.testing.assert_allclose(actual_position, expected_position)
