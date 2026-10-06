import numpy as np
import pytest

from napari._vispy.layers.points import VispyPointsLayer
from napari._vispy.utils.qt_font import FontInfo
from napari.layers import Points


@pytest.mark.parametrize('opacity', [0, 0.3, 0.7, 1])
def test_VispyPointsLayer(opacity):
    points = np.array([[100, 100], [200, 200], [300, 100]])
    layer = Points(points, size=30, opacity=opacity)
    visual = VispyPointsLayer(layer, font_info=FontInfo())
    assert visual.node.opacity == opacity


def test_remove_selected_with_derived_text():
    """See https://github.com/napari/napari/issues/3504"""
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Points(points, text='class', properties=properties)
    vispy_layer = VispyPointsLayer(layer, font_info=FontInfo())
    np.testing.assert_array_equal(vispy_layer.node.text.text, ['A', 'B', 'C'])

    layer.selected_data = {1}
    layer.remove_selected()

    np.testing.assert_array_equal(vispy_layer.node.text.text, ['A', 'C'])


def test_change_text_updates_node_string():
    points = np.random.rand(3, 2)
    properties = {
        'class': np.array(['A', 'B', 'C']),
        'name': np.array(['D', 'E', 'F']),
    }
    layer = Points(points, text='class', properties=properties)
    vispy_layer = VispyPointsLayer(layer, font_info=FontInfo())
    np.testing.assert_array_equal(
        vispy_layer.node.text.text, properties['class']
    )

    layer.text = 'name'

    np.testing.assert_array_equal(
        vispy_layer.node.text.text, properties['name']
    )


def test_change_text_color_updates_node_color():
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    text = {'string': 'class', 'color': [1, 0, 0]}
    layer = Points(points, text=text, properties=properties)
    vispy_layer = VispyPointsLayer(layer, font_info=FontInfo())
    np.testing.assert_array_equal(vispy_layer.node.text.color.rgb, [[1, 0, 0]])

    layer.text.color = [0, 0, 1]

    np.testing.assert_array_equal(vispy_layer.node.text.color.rgb, [[0, 0, 1]])


def test_change_properties_updates_node_strings():
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Points(points, properties=properties, text='class')
    vispy_layer = VispyPointsLayer(layer, font_info=FontInfo())
    np.testing.assert_array_equal(vispy_layer.node.text.text, ['A', 'B', 'C'])

    layer.properties = {'class': np.array(['D', 'E', 'F'])}

    np.testing.assert_array_equal(vispy_layer.node.text.text, ['D', 'E', 'F'])


def test_update_property_value_then_refresh_text_updates_node_strings():
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Points(points, properties=properties, text='class')
    vispy_layer = VispyPointsLayer(layer, font_info=FontInfo())
    np.testing.assert_array_equal(vispy_layer.node.text.text, ['A', 'B', 'C'])

    layer.properties['class'][1] = 'D'
    layer.refresh_text()

    np.testing.assert_array_equal(vispy_layer.node.text.text, ['A', 'D', 'C'])


def test_change_canvas_size_limits():
    points = np.random.rand(3, 2)
    layer = Points(points, canvas_size_limits=(0, 10000))
    vispy_layer = VispyPointsLayer(layer, font_info=FontInfo())
    node = vispy_layer.node

    assert node.canvas_size_limits == (0, 10000)
    layer.canvas_size_limits = (20, 80)
    assert node.canvas_size_limits == (20, 80)


def test_text_with_non_empty_constant_string():
    points = np.random.rand(3, 2)
    layer = Points(points, text={'string': {'constant': 'a'}})

    vispy_layer = VispyPointsLayer(layer, font_info=FontInfo())

    # Vispy cannot broadcast a constant string and assert_array_equal
    # automatically broadcasts, so explicitly check length.
    assert len(vispy_layer.node.text.text) == 3
    np.testing.assert_array_equal(vispy_layer.node.text.text, ['a', 'a', 'a'])

    # Ensure we do position calculation for constants.
    # See https://github.com/napari/napari/issues/5378
    # We want row, column coordinates so drop 3rd dimension and flip.
    actual_position = vispy_layer.node.text.pos[:, 1::-1]
    np.testing.assert_allclose(actual_position, points)


def test_change_antialiasing():
    """Changing antialiasing on the layer should change it on the vispy node."""
    points = np.random.rand(3, 2)
    layer = Points(points)
    vispy_layer = VispyPointsLayer(layer, font_info=FontInfo())
    layer.antialiasing = 5
    assert vispy_layer.node.antialias == layer.antialiasing


@pytest.mark.parametrize('scale', [(-1, -1), (1, -1), (-1, 1)])
def test_negative_scale_highlight(scale):
    """Negative layer scale must not produce negative sizes/widths.

    A negative scale is sometimes used to flip axes (and can be inherited
    from an image layer); vispy rejects negative ``edge_width``, so adding
    or selecting a point used to raise ValueError.
    """
    layer = Points(np.zeros((0, 2)), size=10, scale=scale)
    layer.border_width_is_relative = False
    layer.border_width = 1.0

    vispy_layer = VispyPointsLayer(layer, font_info=FontInfo())

    # previously raised ValueError: edge_width cannot be negative
    layer.add([10, 10])

    for markers in (
        vispy_layer.node.points_markers,
        vispy_layer.node.selection_markers,
    ):
        assert np.all(markers._data['a_size'] > 0)
        assert np.all(markers._data['a_edgewidth'] >= 0)
