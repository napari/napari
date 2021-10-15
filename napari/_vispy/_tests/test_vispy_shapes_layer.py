import numpy as np


def test_change_text_updates_node_string(make_napari_viewer):
    viewer = make_napari_viewer()
    shapes = np.random.rand(3, 4, 2)
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'name': np.array(['D', 'E', 'F']),
    }
    layer = viewer.add_shapes(shapes, properties=properties, text='class')
    vispy_layer = viewer.window.qt_viewer.layer_to_visual[layer]
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, properties['class'])

    layer.text = 'name'

    np.testing.assert_array_equal(text_node.text, properties['name'])


def test_change_text_color_updates_node_color(make_napari_viewer):
    viewer = make_napari_viewer()
    shapes = np.random.rand(3, 4, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    text = {'text': 'class', 'color': [1, 0, 0]}
    layer = viewer.add_shapes(shapes, properties=properties, text=text)
    vispy_layer = viewer.window.qt_viewer.layer_to_visual[layer]
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.color.rgb, [[1, 0, 0]])

    layer.text.color = [0, 0, 1]

    np.testing.assert_array_equal(text_node.color.rgb, [[0, 0, 1]])


def test_change_properties_updates_node_strings(make_napari_viewer):
    viewer = make_napari_viewer()
    shapes = np.random.rand(3, 4, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = viewer.add_shapes(shapes, properties=properties, text='class')
    vispy_layer = viewer.window.qt_viewer.layer_to_visual[layer]
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties = {'class': np.array(['D', 'E', 'F'])}

    np.testing.assert_array_equal(text_node.text, ['D', 'E', 'F'])


def test_update_property_value_then_refresh_text_updates_node_strings(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    shapes = np.random.rand(3, 4, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = viewer.add_shapes(shapes, properties=properties, text='class')
    vispy_layer = viewer.window.qt_viewer.layer_to_visual[layer]
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties['class'][1] = 'D'
    layer.refresh_text()

    np.testing.assert_array_equal(text_node.text, ['A', 'D', 'C'])
