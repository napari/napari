import numpy as np
import pytest


@pytest.mark.parametrize("opacity", [(0), (0.3), (0.7), (1)])
def test_VispyPointsLayer(make_napari_viewer, opacity):
    """Test on the VispyPointsLayer object."""
    viewer = make_napari_viewer()
    points = np.array([[100, 100], [200, 200], [300, 100]])
    layer = viewer.add_points(points, size=30, opacity=opacity)
    visual = viewer.window.qt_viewer.layer_to_visual[layer]
    assert visual.node.opacity == opacity


def test_change_text_updates_node_string(make_napari_viewer):
    viewer = make_napari_viewer()
    points = np.random.rand(3, 2)
    layer = viewer.add_points(points, text='one')
    vispy_layer = viewer.window.qt_viewer.layer_to_visual[layer]
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['one'] * len(points))

    layer.text = 'two'

    np.testing.assert_array_equal(text_node.text, ['two'] * len(points))


def test_change_text_string_updates_node_strings(make_napari_viewer):
    viewer = make_napari_viewer()
    points = np.random.rand(3, 2)
    layer = viewer.add_points(points, text={'text': 'one'})
    vispy_layer = viewer.window.qt_viewer.layer_to_visual[layer]
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['one'] * len(points))

    layer.text.text = 'two'

    np.testing.assert_array_equal(text_node.text, ['two'] * len(points))


def test_change_properties_updates_node_strings(make_napari_viewer):
    viewer = make_napari_viewer()
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = viewer.add_points(points, properties=properties, text='class')
    vispy_layer = viewer.window.qt_viewer.layer_to_visual[layer]
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties = {'class': np.array(['D', 'E', 'F'])}

    np.testing.assert_array_equal(text_node.text, ['D', 'E', 'F'])


def test_update_property_value_then_refresh_text_updates_node_strings(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = viewer.add_points(points, properties=properties, text='class')
    vispy_layer = viewer.window.qt_viewer.layer_to_visual[layer]
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties['class'][1] = 'D'
    layer.refresh_text()

    np.testing.assert_array_equal(text_node.text, ['A', 'D', 'C'])
