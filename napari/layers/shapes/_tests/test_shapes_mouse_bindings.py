import collections

import numpy as np
import pytest

from napari.layers import Shapes
from napari.utils.interactions import (
    ReadOnlyWrapper,
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)


@pytest.fixture
def Event():
    """Create a subclass for simulating vispy mouse events.

    Returns
    -------
    Event : Type
        A new tuple subclass named Event that can be used to create a
        NamedTuple object with fields "type", "is_dragging", and "modifiers".
    """
    return collections.namedtuple(
        'Event', field_names=['type', 'is_dragging', 'modifiers', 'position']
    )


@pytest.fixture
def create_known_shapes_layer():
    """Create shapes layer with known coordinates

    Returns
    -------
    layer : napari.layers.Shapes
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


def test_not_adding_or_selecting_shape(create_known_shapes_layer, Event):
    """Don't add or select a shape by clicking on one in pan_zoom mode."""
    layer, n_shapes, _ = create_known_shapes_layer
    layer.mode = 'pan_zoom'

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=(0, 0),
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=False,
            modifiers=[],
            position=(0, 0),
        )
    )
    mouse_release_callbacks(layer, event)

    # Check no new shape added and non selected
    assert len(layer.data) == n_shapes
    assert len(layer.selected_data) == 0


@pytest.mark.parametrize('shape_type', ['rectangle', 'ellipse', 'line'])
def test_add_simple_shape(shape_type, create_known_shapes_layer, Event):
    """Add simple shape by clicking in add mode."""
    layer, n_shapes, known_non_shape = create_known_shapes_layer

    # Add shape at location where non exists
    layer.mode = 'add_' + shape_type

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=known_non_shape,
        )
    )
    mouse_press_callbacks(layer, event)

    known_non_shape_end = [40, 60]
    # Simulate drag end
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            modifiers=[],
            position=known_non_shape_end,
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=False,
            modifiers=[],
            position=known_non_shape_end,
        )
    )
    mouse_release_callbacks(layer, event)

    # Check new shape added at coordinates
    assert len(layer.data) == n_shapes + 1
    np.testing.assert_allclose(layer.data[-1][0], known_non_shape)
    new_shape_max = np.max(layer.data[-1], axis=0)
    np.testing.assert_allclose(new_shape_max, known_non_shape_end)
    assert layer.shape_type[-1] == shape_type


@pytest.mark.parametrize('shape_type', ['path', 'polygon'])
def test_add_complex_shape(shape_type, create_known_shapes_layer, Event):
    """Add simple shape by clicking in add mode."""
    layer, n_shapes, known_non_shape = create_known_shapes_layer

    desired_shape = [[20, 30], [10, 50], [60, 40], [80, 20]]
    # Add shape at location where non exists
    layer.mode = 'add_' + shape_type

    for coord in desired_shape:
        # Simulate move, click, and release
        event = ReadOnlyWrapper(
            Event(
                type='mouse_move',
                is_dragging=False,
                modifiers=[],
                position=coord,
            )
        )
        mouse_move_callbacks(layer, event)
        event = ReadOnlyWrapper(
            Event(
                type='mouse_press',
                is_dragging=False,
                modifiers=[],
                position=coord,
            )
        )
        mouse_press_callbacks(layer, event)
        event = ReadOnlyWrapper(
            Event(
                type='mouse_release',
                is_dragging=False,
                modifiers=[],
                position=coord,
            )
        )
        mouse_release_callbacks(layer, event)

    # finish drawing
    layer._finish_drawing()

    # Check new shape added at coordinates
    assert len(layer.data) == n_shapes + 1
    np.testing.assert_allclose(layer.data[-1], desired_shape)
    assert layer.shape_type[-1] == shape_type


def test_vertex_insert(create_known_shapes_layer, Event):
    """Add vertex to shape."""
    layer, n_shapes, known_non_shape = create_known_shapes_layer

    n_coord = len(layer.data[0])
    layer.mode = 'vertex_insert'
    layer.selected_data = {0}

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=known_non_shape,
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag end
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            modifiers=[],
            position=known_non_shape,
        )
    )
    mouse_move_callbacks(layer, event)

    # Check new shape added at coordinates
    assert len(layer.data) == n_shapes
    assert len(layer.data[0]) == n_coord + 1
    np.testing.assert_allclose(
        np.min(abs(layer.data[0] - known_non_shape), axis=0), [0, 0]
    )


def test_vertex_remove(create_known_shapes_layer, Event):
    """Remove vertex from shape."""
    layer, n_shapes, known_non_shape = create_known_shapes_layer

    n_coord = len(layer.data[0])
    layer.mode = 'vertex_remove'
    layer.selected_data = {0}
    position = tuple(layer.data[0][0])

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag end
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            modifiers=[],
            position=position,
        )
    )
    mouse_move_callbacks(layer, event)

    # Check new shape added at coordinates
    assert len(layer.data) == n_shapes
    assert len(layer.data[0]) == n_coord - 1


@pytest.mark.parametrize('mode', ['select', 'direct'])
def test_select_shape(mode, create_known_shapes_layer, Event):
    """Select a shape by clicking on one in select mode."""
    layer, n_shapes, _ = create_known_shapes_layer

    layer.mode = mode
    position = tuple(layer.data[0][0])

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
    )
    mouse_release_callbacks(layer, event)

    # Check clicked shape selected
    assert len(layer.selected_data) == 1
    assert layer.selected_data == {0}


def test_drag_shape(create_known_shapes_layer, Event):
    """Select and drag vertex."""
    layer, n_shapes, _ = create_known_shapes_layer

    layer.mode = 'select'
    # Zoom in so as to not select any vertices
    layer.scale_factor = 0.01
    orig_data = layer.data[0].copy()
    assert len(layer.selected_data) == 0

    position = tuple(np.mean(layer.data[0], axis=0))

    # Check shape under cursor
    value = layer.get_value(position, world=True)
    assert value == (0, None)

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
    )
    mouse_press_callbacks(layer, event)
    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
    )
    mouse_release_callbacks(layer, event)

    assert len(layer.selected_data) == 1
    assert layer.selected_data == {0}

    # Check shape but not vertex under cursor
    value = layer.get_value(event.position, world=True)
    assert value == (0, None)

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=True,
            modifiers=[],
            position=position,
        )
    )
    mouse_press_callbacks(layer, event)
    # start drag event
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            modifiers=[],
            position=position,
        )
    )
    mouse_move_callbacks(layer, event)
    position = tuple(np.add(position, [10, 5]))
    # Simulate move, click, and release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            modifiers=[],
            position=position,
        )
    )
    mouse_move_callbacks(layer, event)
    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=True,
            modifiers=[],
            position=position,
        )
    )
    mouse_release_callbacks(layer, event)

    # Check clicked shape selected
    assert len(layer.selected_data) == 1
    assert layer.selected_data == {0}
    np.testing.assert_allclose(layer.data[0], orig_data + [10, 5])


def test_drag_vertex(create_known_shapes_layer, Event):
    """Select and drag vertex."""
    layer, n_shapes, _ = create_known_shapes_layer

    layer.mode = 'direct'
    layer.selected_data = {0}
    position = tuple(layer.data[0][0])

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
    )
    mouse_press_callbacks(layer, event)

    position = [0, 0]
    # Simulate move, click, and release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            modifiers=[],
            position=position,
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=True,
            modifiers=[],
            position=position,
        )
    )
    mouse_release_callbacks(layer, event)

    # Check clicked shape selected
    assert len(layer.selected_data) == 1
    assert layer.selected_data == {0}
    np.testing.assert_allclose(layer.data[0][-1], [0, 0])


@pytest.mark.parametrize(
    'mode',
    [
        'select',
        'direct',
        'add_rectangle',
        'add_ellipse',
        'add_line',
        'add_polygon',
        'add_path',
        'vertex_insert',
        'vertex_remove',
    ],
)
def test_after_in_add_mode_shape(mode, create_known_shapes_layer, Event):
    """Don't add or select a shape by clicking on one in pan_zoom mode."""
    layer, n_shapes, _ = create_known_shapes_layer

    layer.mode = mode
    layer.mode = 'pan_zoom'
    position = tuple(layer.data[0][0])

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
    )
    mouse_release_callbacks(layer, event)

    # Check no new shape added and non selected
    assert len(layer.data) == n_shapes
    assert len(layer.selected_data) == 0


@pytest.mark.parametrize('mode', ['select', 'direct'])
def test_unselect_select_shape(mode, create_known_shapes_layer, Event):
    """Select a shape by clicking on one in select mode."""
    layer, n_shapes, _ = create_known_shapes_layer

    layer.mode = mode
    position = tuple(layer.data[0][0])
    layer.selected_data = {1}

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
    )
    mouse_release_callbacks(layer, event)

    # Check clicked shape selected
    assert len(layer.selected_data) == 1
    assert layer.selected_data == {0}


@pytest.mark.parametrize('mode', ['select', 'direct'])
def test_not_selecting_shape(mode, create_known_shapes_layer, Event):
    """Don't select a shape by not clicking on one in select mode."""
    layer, n_shapes, known_non_shape = create_known_shapes_layer

    layer.mode = mode

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=known_non_shape,
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=False,
            modifiers=[],
            position=known_non_shape,
        )
    )
    mouse_release_callbacks(layer, event)

    # Check clicked shape selected
    assert len(layer.selected_data) == 0


@pytest.mark.parametrize('mode', ['select', 'direct'])
def test_unselecting_shapes(mode, create_known_shapes_layer, Event):
    """Unselect shapes by not clicking on one in select mode."""
    layer, n_shapes, known_non_shape = create_known_shapes_layer

    layer.mode = mode
    layer.selected_data = {0, 1}
    assert len(layer.selected_data) == 2

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=known_non_shape,
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=False,
            modifiers=[],
            position=known_non_shape,
        )
    )
    mouse_release_callbacks(layer, event)

    # Check clicked shape selected
    assert len(layer.selected_data) == 0


@pytest.mark.parametrize('mode', ['select', 'direct'])
def test_selecting_shapes_with_drag(mode, create_known_shapes_layer, Event):
    """Select all shapes when drag box includes all of them."""
    layer, n_shapes, known_non_shape = create_known_shapes_layer

    layer.mode = mode

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=known_non_shape,
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag start
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            modifiers=[],
            position=known_non_shape,
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate drag end
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move', is_dragging=True, modifiers=[], position=(0, 0)
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=True,
            modifiers=[],
            position=(0, 0),
        )
    )
    mouse_release_callbacks(layer, event)

    # Check all shapes selected as drag box contains them
    assert len(layer.selected_data) == n_shapes


@pytest.mark.parametrize('mode', ['select', 'direct'])
def test_selecting_no_shapes_with_drag(mode, create_known_shapes_layer, Event):
    """Select all shapes when drag box includes all of them."""
    layer, n_shapes, known_non_shape = create_known_shapes_layer

    layer.mode = mode

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=known_non_shape,
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag start
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            modifiers=[],
            position=known_non_shape,
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate drag end
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            modifiers=[],
            position=(50, 60),
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=True,
            modifiers=[],
            position=(50, 60),
        )
    )
    mouse_release_callbacks(layer, event)

    # Check no shapes selected as drag box doesn't contain them
    assert len(layer.selected_data) == 0
