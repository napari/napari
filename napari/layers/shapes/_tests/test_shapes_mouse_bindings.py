import numpy as np
import pytest
import collections
from napari.layers import Shapes
from napari.utils.interactions import (
    ReadOnlyWrapper,
    mouse_press_callbacks,
    mouse_move_callbacks,
    mouse_release_callbacks,
)


def create_known_shapes_layer():
    """Create shapes layer with known coordinates

    Returns
    -------
    layer : napar.layers.Shapes
        Shapes layer.
    n_shapes : int
        Number of shapes in the shapes layer
    known_non_shape : list
        Data coordinates that are known to contain no shapes. Useful during
        testing when needing to guarantee no shape is clicked on.
    """
    data = [[[1, 3], [8, 4]], [[10, 10], [15, 4]]]
    known_non_shape = [20, 30]
    n_shapes = len(data)

    layer = Shapes(data)
    assert layer.ndim == 2
    assert len(layer.data) == n_shapes
    assert len(layer.selected_data) == 0

    return layer, n_shapes, known_non_shape


def test_not_adding_or_selecting_shape():
    """Don't add or select a shape by clicking on one in pan_zoom mode."""
    layer, n_shapes, _ = create_known_shapes_layer()
    layer.mode = 'pan_zoom'

    Event = collections.namedtuple('Event', 'type is_dragging')

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(Event(type='mouse_release', is_dragging=False))
    mouse_release_callbacks(layer, event)

    # Check no new shape added and non selected
    assert len(layer.data) == n_shapes
    assert len(layer.selected_data) == 0


@pytest.mark.parametrize('shape_type', ['rectangle', 'ellipse', 'line'])
def test_add_shape(shape_type):
    """Add shape by clicking in add mode."""
    layer, n_shapes, known_non_shape = create_known_shapes_layer()

    # Add shape at location where non exists
    layer.mode = 'add_' + shape_type
    layer.position = known_non_shape

    Event = collections.namedtuple('Event', 'type is_dragging')

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
    mouse_press_callbacks(layer, event)

    known_non_shape_end = [40, 60]
    layer.position = known_non_shape_end
    # Simulate drag end
    event = ReadOnlyWrapper(Event(type='mouse_move', is_dragging=True))
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(Event(type='mouse_release', is_dragging=False))
    mouse_release_callbacks(layer, event)

    # Check new rectangle added at coordinates location
    assert len(layer.data) == n_shapes + 1
    np.testing.assert_allclose(layer.data[-1][0], known_non_shape)
    new_shape_max = np.max(layer.data[-1], axis=0)
    np.testing.assert_allclose(new_shape_max, known_non_shape_end)
    assert layer.shape_type[-1] == shape_type


# def test_select_shape():
#     """Select a shape by clicking on one in select mode."""
#     layer, n_shapes, _ = create_known_shapes_layer()
#
#     layer.mode = 'select'
#     layer.position = tuple(layer.data[0])
#
#     Event = collections.namedtuple('Event', 'type is_dragging modifiers')
#
#     # Simulate click
#     event = ReadOnlyWrapper(
#         Event(type='mouse_press', is_dragging=False, modifiers=[])
#     )
#     mouse_press_callbacks(layer, event)
#
#     # Simulate release
#     event = ReadOnlyWrapper(
#         Event(type='mouse_release', is_dragging=False, modifiers=[])
#     )
#     mouse_release_callbacks(layer, event)
#
#     # Check clicked shape selected
#     assert len(layer.selected_data) == 1
#     assert layer.selected_data[0] == 0
#
#
# def test_not_adding_or_selecting_after_in_add_mode_shape():
#     """Don't add or select a shape by clicking on one in pan_zoom mode."""
#     layer, n_shapes, _ = create_known_shapes_layer()
#
#     layer.mode = 'add'
#     layer.mode = 'pan_zoom'
#     layer.position = tuple(layer.data[0])
#
#     Event = collections.namedtuple('Event', 'type is_dragging')
#
#     # Simulate click
#     event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
#     mouse_press_callbacks(layer, event)
#
#     # Simulate release
#     event = ReadOnlyWrapper(Event(type='mouse_release', is_dragging=False))
#     mouse_release_callbacks(layer, event)
#
#     # Check no new shape added and non selected
#     assert len(layer.data) == n_shapes
#     assert len(layer.selected_data) == 0
#
#
# def test_not_adding_or_selecting_after_in_select_mode_shape():
#     """Don't add or select a shape by clicking on one in pan_zoom mode."""
#     layer, n_shapes, _ = create_known_shapes_layer()
#
#     layer.mode = 'select'
#     layer.mode = 'pan_zoom'
#     layer.position = tuple(layer.data[0])
#
#     Event = collections.namedtuple('Event', 'type is_dragging')
#
#     # Simulate click
#     event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
#     mouse_press_callbacks(layer, event)
#
#     # Simulate release
#     event = ReadOnlyWrapper(Event(type='mouse_release', is_dragging=False))
#     mouse_release_callbacks(layer, event)
#
#     # Check no new shape added and non selected
#     assert len(layer.data) == n_shapes
#     assert len(layer.selected_data) == 0
#
#
# def test_unselect_select_shape():
#     """Select a shape by clicking on one in select mode."""
#     layer, n_shapes, _ = create_known_shapes_layer()
#
#     layer.mode = 'select'
#     layer.position = tuple(layer.data[0])
#     layer.selected_data = [2, 3]
#
#     Event = collections.namedtuple('Event', 'type is_dragging modifiers')
#
#     # Simulate click
#     event = ReadOnlyWrapper(
#         Event(type='mouse_press', is_dragging=False, modifiers=[])
#     )
#     mouse_press_callbacks(layer, event)
#
#     # Simulate release
#     event = ReadOnlyWrapper(
#         Event(type='mouse_release', is_dragging=False, modifiers=[])
#     )
#     mouse_release_callbacks(layer, event)
#
#     # Check clicked shape selected
#     assert len(layer.selected_data) == 1
#     assert layer.selected_data[0] == 0
#
#
# def test_add_select_shape():
#     """Add to a selection of shapes shape by shift-clicking on one."""
#     layer, n_shapes, _ = create_known_shapes_layer()
#
#     layer.mode = 'select'
#     layer.position = tuple(layer.data[0])
#     layer.selected_data = [2, 3]
#
#     Event = collections.namedtuple('Event', 'type is_dragging modifiers')
#
#     # Simulate click
#     event = ReadOnlyWrapper(
#         Event(type='mouse_press', is_dragging=False, modifiers=['Shift'])
#     )
#     mouse_press_callbacks(layer, event)
#
#     # Simulate release
#     event = ReadOnlyWrapper(
#         Event(type='mouse_release', is_dragging=False, modifiers=['Shift'])
#     )
#     mouse_release_callbacks(layer, event)
#
#     # Check clicked shape selected
#     assert len(layer.selected_data) == 3
#     assert layer.selected_data == [2, 3, 0]
#
#
# def test_remove_select_shape():
#     """Remove from a selection of shapes shape by shift-clicking on one."""
#     layer, n_shapes, _ = create_known_shapes_layer()
#
#     layer.mode = 'select'
#     layer.position = tuple(layer.data[0])
#     layer.selected_data = [0, 2, 3]
#
#     Event = collections.namedtuple('Event', 'type is_dragging modifiers')
#
#     # Simulate click
#     event = ReadOnlyWrapper(
#         Event(type='mouse_press', is_dragging=False, modifiers=['Shift'])
#     )
#     mouse_press_callbacks(layer, event)
#
#     # Simulate release
#     event = ReadOnlyWrapper(
#         Event(type='mouse_release', is_dragging=False, modifiers=['Shift'])
#     )
#     mouse_release_callbacks(layer, event)
#
#     # Check clicked shape selected
#     assert len(layer.selected_data) == 2
#     assert layer.selected_data == [2, 3]
#
#
# def test_not_selecting_shape():
#     """Don't select a shape by not clicking on one in select mode."""
#     layer, n_shapes, known_non_shape = create_known_shapes_layer()
#
#     layer.mode = 'select'
#     layer.position = known_non_shape
#
#     Event = collections.namedtuple('Event', 'type is_dragging modifiers')
#
#     # Simulate click
#     event = ReadOnlyWrapper(
#         Event(type='mouse_press', is_dragging=False, modifiers=[])
#     )
#     mouse_press_callbacks(layer, event)
#
#     # Simulate release
#     event = ReadOnlyWrapper(
#         Event(type='mouse_release', is_dragging=False, modifiers=[])
#     )
#     mouse_release_callbacks(layer, event)
#
#     # Check clicked shape selected
#     assert len(layer.selected_data) == 0
#
#
# def test_unselecting_shapes():
#     """Unselect shapes by not clicking on one in select mode."""
#     layer, n_shapes, known_non_shape = create_known_shapes_layer()
#
#     layer.mode = 'select'
#     layer.position = known_non_shape
#     layer.selected_data = [2, 3]
#     assert len(layer.selected_data) == 2
#
#     Event = collections.namedtuple('Event', 'type is_dragging modifiers')
#
#     # Simulate click
#     event = ReadOnlyWrapper(
#         Event(type='mouse_press', is_dragging=False, modifiers=[])
#     )
#     mouse_press_callbacks(layer, event)
#
#     # Simulate release
#     event = ReadOnlyWrapper(
#         Event(type='mouse_release', is_dragging=False, modifiers=[])
#     )
#     mouse_release_callbacks(layer, event)
#
#     # Check clicked shape selected
#     assert len(layer.selected_data) == 0
#
#
# def test_selecting_all_shapes_with_drag():
#     """Select all shapes when drag box includes all of them."""
#     layer, n_shapes, known_non_shape = create_known_shapes_layer()
#
#     layer.mode = 'select'
#     layer.position = known_non_shape
#
#     Event = collections.namedtuple('Event', 'type is_dragging modifiers')
#
#     # Simulate click
#     event = ReadOnlyWrapper(
#         Event(type='mouse_press', is_dragging=False, modifiers=[])
#     )
#     mouse_press_callbacks(layer, event)
#
#     # Simulate drag start
#     event = ReadOnlyWrapper(
#         Event(type='mouse_move', is_dragging=True, modifiers=[])
#     )
#     mouse_move_callbacks(layer, event)
#
#     layer.position = [0, 0]
#     # Simulate drag end
#     event = ReadOnlyWrapper(
#         Event(type='mouse_move', is_dragging=True, modifiers=[])
#     )
#     mouse_move_callbacks(layer, event)
#
#     # Simulate release
#     event = ReadOnlyWrapper(
#         Event(type='mouse_release', is_dragging=True, modifiers=[])
#     )
#     mouse_release_callbacks(layer, event)
#
#     # Check all shapes selected as drag box contains them
#     assert len(layer.selected_data) == n_shapes
#
#
# def test_selecting_no_shapes_with_drag():
#     """Select all shapes when drag box includes all of them."""
#     layer, n_shapes, known_non_shape = create_known_shapes_layer()
#
#     layer.mode = 'select'
#     layer.position = known_non_shape
#
#     Event = collections.namedtuple('Event', 'type is_dragging modifiers')
#
#     # Simulate click
#     event = ReadOnlyWrapper(
#         Event(type='mouse_press', is_dragging=False, modifiers=[])
#     )
#     mouse_press_callbacks(layer, event)
#
#     # Simulate drag start
#     event = ReadOnlyWrapper(
#         Event(type='mouse_move', is_dragging=True, modifiers=[])
#     )
#     mouse_move_callbacks(layer, event)
#
#     layer.position = [50, 60]
#     # Simulate drag end
#     event = ReadOnlyWrapper(
#         Event(type='mouse_move', is_dragging=True, modifiers=[])
#     )
#     mouse_move_callbacks(layer, event)
#
#     # Simulate release
#     event = ReadOnlyWrapper(
#         Event(type='mouse_release', is_dragging=True, modifiers=[])
#     )
#     mouse_release_callbacks(layer, event)
#
#     # Check no shapes selected as drag box doesn't contain them
#     assert len(layer.selected_data) == 0
