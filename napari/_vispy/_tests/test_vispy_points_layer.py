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
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'name': np.array(['D', 'E', 'F']),
    }
    layer = viewer.add_points(points, text='class', properties=properties)
    vispy_layer = viewer.window.qt_viewer.layer_to_visual[layer]
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, properties['class'])

    layer.text = 'name'

    np.testing.assert_array_equal(text_node.text, properties['name'])


def test_change_text_color_updates_node_colors(make_napari_viewer):
    viewer = make_napari_viewer()
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    text = {'text': 'class', 'color': [1, 0, 0]}
    layer = viewer.add_points(points, text=text, properties=properties)
    vispy_layer = viewer.window.qt_viewer.layer_to_visual[layer]
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.color.rgb, [[1, 0, 0]])

    layer.text.color = [0, 0, 1]

    np.testing.assert_array_equal(text_node.color.rgb, [[0, 0, 1]])
