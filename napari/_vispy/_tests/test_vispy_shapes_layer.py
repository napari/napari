import numpy as np

from napari._vispy.layers.shapes import VispyShapesLayer
from napari.layers import Shapes


def test_change_text_updates_node_string():
    shapes = np.random.rand(3, 4, 2)
    layer = Shapes(shapes, text='one')
    vispy_layer = VispyShapesLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['one'] * len(shapes))

    layer.text = 'two'

    np.testing.assert_array_equal(text_node.text, ['two'] * len(shapes))


def test_change_text_string_updates_node_strings():
    shapes = np.random.rand(3, 4, 2)
    layer = Shapes(shapes, text={'text': 'one'})
    vispy_layer = VispyShapesLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['one'] * len(shapes))

    layer.text.text = 'two'

    np.testing.assert_array_equal(text_node.text, ['two'] * len(shapes))


def test_change_text_color_updates_node_colors():
    shapes = np.random.rand(3, 4, 2)
    layer = Shapes(shapes, text={'color': [1, 0, 0]})
    vispy_layer = VispyShapesLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(
        text_node.color.rgb, [[1, 0, 0]] * len(shapes)
    )

    layer.text.color = [0, 0, 1]

    np.testing.assert_array_equal(
        text_node.color.rgb, [[0, 0, 1]] * len(shapes)
    )


def test_change_properties_updates_node_strings():
    shapes = np.random.rand(3, 4, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Shapes(shapes, properties=properties, text='class')
    vispy_layer = VispyShapesLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties = {'class': np.array(['D', 'E', 'F'])}

    np.testing.assert_array_equal(text_node.text, ['D', 'E', 'F'])


def test_update_property_value_then_refresh_text_updates_node_strings():
    shapes = np.random.rand(3, 4, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Shapes(shapes, properties=properties, text='class')
    vispy_layer = VispyShapesLayer(layer)
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties['class'][1] = 'D'
    layer.refresh_text()

    np.testing.assert_array_equal(text_node.text, ['A', 'D', 'C'])
