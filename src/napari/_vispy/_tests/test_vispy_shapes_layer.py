import numpy as np

from napari._vispy.layers.shapes import VispyShapesLayer
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.layers import Shapes
from napari.utils._test_utils import read_only_mouse_event
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)


def test_active_shape_overlay_tracks_staged_geometry():
    viewer = ViewerModel()
    layer = viewer.add_shapes([np.array([[0, 0], [0, 2], [2, 2], [2, 0]])])
    vispy_layer = VispyShapesLayer(layer, font_info=FontInfo())
    assert vispy_layer.node._subvisuals[0] is vispy_layer.node.shape_faces
    assert vispy_layer.node._subvisuals[1] is vispy_layer.node.shape_highlights
    assert vispy_layer.node._subvisuals[2] is vispy_layer.node.highlight_lines
    layer.mode = 'add_rectangle'

    mouse_press_callbacks(
        layer,
        read_only_mouse_event(type='mouse_press', position=[5, 5]),
    )
    mouse_move_callbacks(
        layer,
        read_only_mouse_event(
            type='mouse_move', is_dragging=True, position=[10, 12]
        ),
    )

    vertices = vispy_layer.node.shape_highlights.mesh_data.get_vertices()
    index = layer._data_view.staged_index
    assert index is not None
    shape = layer._data_view.shapes[index]
    expected_vertices = np.concatenate(
        [
            shape._face_vertices,
            shape._edge_vertices + shape.edge_width * shape._edge_offsets,
        ]
    )[:, ::-1]
    np.testing.assert_allclose(
        vertices[: len(expected_vertices)], expected_vertices
    )

    mouse_release_callbacks(
        layer,
        read_only_mouse_event(type='mouse_release', position=[10, 12]),
    )

    outline_vertices, outline_faces = layer._outline_shapes()
    np.testing.assert_allclose(
        vispy_layer.node.shape_highlights.mesh_data.get_vertices(),
        outline_vertices,
    )
    np.testing.assert_array_equal(
        vispy_layer.node.shape_highlights.mesh_data.get_faces(), outline_faces
    )


def test_remove_selected_with_derived_text():
    """See https://github.com/napari/napari/issues/3504"""
    np.random.seed(0)
    shapes = np.random.rand(3, 4, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Shapes(shapes, properties=properties, text='class')
    vispy_layer = VispyShapesLayer(layer, font_info=FontInfo())
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
    vispy_layer = VispyShapesLayer(layer, font_info=FontInfo())
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
    vispy_layer = VispyShapesLayer(layer, font_info=FontInfo())
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.color.rgb, [[1, 0, 0]])

    layer.text.color = [0, 0, 1]

    np.testing.assert_array_equal(text_node.color.rgb, [[0, 0, 1]])


def test_change_properties_updates_node_strings():
    np.random.seed(0)
    shapes = np.random.rand(3, 4, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Shapes(shapes, properties=properties, text='class')
    vispy_layer = VispyShapesLayer(layer, font_info=FontInfo())
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties = {'class': np.array(['D', 'E', 'F'])}

    np.testing.assert_array_equal(text_node.text, ['D', 'E', 'F'])


def test_update_property_value_then_refresh_text_updates_node_strings():
    np.random.seed(0)
    shapes = np.random.rand(3, 4, 2)
    properties = {'class': np.array(['A', 'B', 'C'])}
    layer = Shapes(shapes, properties=properties, text='class')
    vispy_layer = VispyShapesLayer(layer, font_info=FontInfo())
    text_node = vispy_layer._get_text_node()
    np.testing.assert_array_equal(text_node.text, ['A', 'B', 'C'])

    layer.properties['class'][1] = 'D'
    layer.refresh_text()

    np.testing.assert_array_equal(text_node.text, ['A', 'D', 'C'])


def test_text_with_non_empty_constant_string():
    np.random.seed(0)
    shapes = np.random.rand(3, 4, 2)
    layer = Shapes(shapes, text={'string': {'constant': 'a'}})

    vispy_layer = VispyShapesLayer(layer, font_info=FontInfo())

    text_node = vispy_layer._get_text_node()
    # Vispy cannot broadcast a constant string and assert_array_equal
    # automatically broadcasts, so explicitly check length.
    assert len(text_node.text) == 3
    np.testing.assert_array_equal(text_node.text, ['a', 'a', 'a'])

    # Ensure we do position calculation for constants.
    # See https://github.com/napari/napari/issues/5378
    expected_position = np.mean(shapes, axis=1)
    # We want row, column coordinates so drop 3rd dimension and flip.
    actual_position = text_node.pos[:, 1::-1]
    np.testing.assert_allclose(actual_position, expected_position)
