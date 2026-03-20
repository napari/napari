import numpy as np
import pytest

from napari._vispy.layers.points import VispyPointsLayer
from napari.components import Dims
from napari.layers import Points


@pytest.mark.parametrize('opacity', [0, 0.3, 0.7, 1])
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
    vispy_layer = VispyPointsLayer(layer)
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
    vispy_layer = VispyPointsLayer(layer)
    np.testing.assert_array_equal(vispy_layer.node.text.color.rgb, [[1, 0, 0]])

    layer.text.color = [0, 0, 1]

    np.testing.assert_array_equal(vispy_layer.node.text.color.rgb, [[0, 0, 1]])


def test_change_properties_updates_node_strings():
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Points(points, properties=properties, text='class')
    vispy_layer = VispyPointsLayer(layer)
    np.testing.assert_array_equal(vispy_layer.node.text.text, ['A', 'B', 'C'])

    layer.properties = {'class': np.array(['D', 'E', 'F'])}

    np.testing.assert_array_equal(vispy_layer.node.text.text, ['D', 'E', 'F'])


def test_update_property_value_then_refresh_text_updates_node_strings():
    points = np.random.rand(3, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Points(points, properties=properties, text='class')
    vispy_layer = VispyPointsLayer(layer)
    np.testing.assert_array_equal(vispy_layer.node.text.text, ['A', 'B', 'C'])

    layer.properties['class'][1] = 'D'
    layer.refresh_text()

    np.testing.assert_array_equal(vispy_layer.node.text.text, ['A', 'D', 'C'])


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
    vispy_layer = VispyPointsLayer(layer)
    layer.antialiasing = 5
    assert vispy_layer.node.antialias == layer.antialiasing


def test_highlight_with_out_of_slice_display():
    """Highlight should work when out_of_slice_display is enabled.

    Regression test for a bug where _view_size_scale (array for all view
    points) was multiplied with size indexed only by highlighted points,
    causing a shape mismatch when more than one point was in view but only
    a subset was highlighted.
    """
    # Place 5 points at known z positions, with a large size so all 5 spill
    # into the z=50 slice and _view_size_scale becomes a (5,) array.
    data = np.array(
        [[0, 0, 0], [25, 0, 0], [50, 0, 0], [75, 0, 0], [100, 0, 0]],
        dtype=float,
    )
    layer = Points(data, size=200)
    vispy_layer = VispyPointsLayer(layer)

    # Select point 0 BEFORE slicing so update_selected_view populates
    # _selected_view and _set_highlight populates _highlight_index.
    layer.selected_data = {0}
    layer.out_of_slice_display = True
    layer._slice_dims(Dims(ndim=3, point=(50, 0, 0)))

    # Verify the preconditions that cause the bug:
    # all 5 points in view, scale is a per-point array, only 1 highlighted
    assert len(layer._view_indices) == 5
    assert isinstance(layer._view_size_scale, np.ndarray)
    assert len(layer._highlight_index) == 1

    # Previously, raised ValueError: could not broadcast input array from shape (5,) into shape (1,)
    vispy_layer._on_highlight_change()
