import numpy as np

from napari._vispy.layers.shapes import VispyShapesLayer
from napari.layers import Shapes


def test_remove_selected_with_derived_text():
    """See https://github.com/napari/napari/issues/3504"""
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


def test_text_with_non_empty_constant_string():
    shapes = np.random.rand(3, 4, 2)
    layer = Shapes(shapes, text={'string': {'constant': 'a'}})

    vispy_layer = VispyShapesLayer(layer)

    text_node = vispy_layer._get_text_node()
    # Vispy cannot broadcast a constant string and assert_array_equal
    # automatically broadcasts, so explicitly check length.
    assert len(text_node.text) == 3
    np.testing.assert_array_equal(text_node.text, ['a', 'a', 'a'])


def test_text_with_non_empty_constant_string_alt():
    num_shapes = 3
    bar_len = 200
    lines = np.array(
        [[[i, 400, 100], [i, 400, 100 + bar_len]] for i in range(num_shapes)]
    )

    layer = Shapes(
        lines,
        shape_type='line',
        name='scale bar',
        features={'bar_len': np.ones(3) * bar_len},
        text={
            'string': {'constant': '200'},
            'size': 30,
            'color': 'red',
            'translation': np.array([0, 5, 0]),
        },
        edge_width=2,
        edge_color=[1, 0, 0, 1],
        face_color=[0, 0, 0, 0],
    )

    vispy_layer = VispyShapesLayer(layer)
    text_node = vispy_layer._get_text_node()

    # Vispy cannot broadcast a constant string and assert_array_equal
    # automatically broadcasts, so explicitly check length.
    assert len(text_node.text) == 1
    np.testing.assert_array_equal(text_node.text, ['200'])

    np.testing.assert_array_equal(text_node.pos, [[200, 405]])
