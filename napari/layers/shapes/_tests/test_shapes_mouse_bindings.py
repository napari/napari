import collections

import numpy as np
import pytest

from napari.layers import Shapes
from napari.layers.shapes.shapes import Mode
from napari.utils._proxies import ReadOnlyWrapper
from napari.utils.interactions import (
    mouse_double_click_callbacks,
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
    end_click = ReadOnlyWrapper(
        Event(
            type='mouse_double_click',
            is_dragging=False,
            modifiers=[],
            position=coord,
        )
    )
    assert layer.mouse_double_click_callbacks
    mouse_double_click_callbacks(layer, end_click)

    # Check new shape added at coordinates
    assert len(layer.data) == n_shapes + 1
    assert layer.data[-1].shape, desired_shape.shape
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


@pytest.mark.parametrize(
    'mode',
    [
        'add_polygon',
        'add_path',
    ],
)
def test_clicking_the_same_point_is_not_crashing(mode, create_known_shapes_layer, Event):
    layer, n_shapes, _ = create_known_shapes_layer

    layer.mode = mode
    position = tuple(layer.data[0][0])

    for _ in range(2):
        event = ReadOnlyWrapper(
            Event(
                type='mouse_press',
                is_dragging=False,
                modifiers=[],
                position=position,
            )
        )
        mouse_press_callbacks(layer, event)

        event = ReadOnlyWrapper(
            Event(
                type='mouse_release',
                is_dragging=False,
                modifiers=[],
                position=position,
            )
        )
        mouse_release_callbacks(layer, event)


@pytest.mark.parametrize(
    'mode',
    [
        'add_polygon',
        'add_path',
    ],
)
def test_is_creating_is_false_on_creation(mode, create_known_shapes_layer, Event):
    layer, n_shapes, _ = create_known_shapes_layer

    layer.mode = mode
    position = tuple(layer.data[0][0])

    def is_creating_is_True(event):
        assert event.source._is_creating

    def is_creating_is_False(event):
        assert not event.source._is_creating


    assert not layer._is_creating
    layer.events.set_data.connect(is_creating_is_True)

    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
    )
    mouse_press_callbacks(layer, event)

    assert layer._is_creating

    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
    )
    mouse_release_callbacks(layer, event)

    assert layer._is_creating

    layer.events.set_data.disconnect(is_creating_is_True)
    layer.events.set_data.connect(is_creating_is_False)
    end_click = ReadOnlyWrapper(
        Event(
            type='mouse_double_click',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
    )
    mouse_double_click_callbacks(layer, end_click)

    assert not layer._is_creating


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


@pytest.mark.parametrize(
    'attr', ('_move_modes', '_drag_modes', '_cursor_modes')
)
def test_all_modes_covered(attr):
    """
    Test that all dictionaries modes have all the keys, this simplify the handling logic
    As we do not need to test whether a key is in a dict or not.
    """
    mode_dict = getattr(Shapes, attr)
    assert {k.value for k in mode_dict.keys()} == set(Mode.keys())


@pytest.mark.parametrize(
    'pre_selection,on_point,modifier',
    [
        (set(), True, []),
        ({1}, True, []),
    ],
)
def test_drag_start_selection(
    create_known_shapes_layer, Event, pre_selection, on_point, modifier
):
    """Check layer drag start and drag box behave as expected."""
    layer, n_points, known_non_point = create_known_shapes_layer
    layer.mode = 'select'
    layer.selected_data = pre_selection

    if on_point:
        initial_position = tuple(layer.data[0].mean(axis=0))
    else:
        initial_position = tuple(known_non_point)
    zero_pos = [0, 0]

    value = layer.get_value(initial_position, world=True)
    assert value[0] == 0
    assert layer._drag_start is None
    assert layer._drag_box is None
    assert layer.selected_data == pre_selection

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            position=initial_position,
            modifiers=modifier,
            is_dragging=True,
        )
    )
    mouse_press_callbacks(layer, event)

    if modifier:
        if not on_point:
            assert layer.selected_data == pre_selection
        elif 0 in pre_selection:
            assert layer.selected_data == pre_selection - {0}
        else:
            assert layer.selected_data == pre_selection | {0}
    elif not on_point:
        assert layer.selected_data == set()
    elif 0 in pre_selection:
        assert layer.selected_data == pre_selection
    else:
        assert layer.selected_data == {0}

    if len(layer.selected_data) > 0:
        center_list = []
        for idx in layer.selected_data:
            center_list.append(layer.data[idx].mean(axis=0))
        center = np.mean(center_list, axis=0)
    else:
        center = [0, 0]

    if not modifier:
        start_position = [
            initial_position[0] - center[0],
            initial_position[1] - center[1],
        ]
    else:
        start_position = initial_position

    is_point_move = len(layer.selected_data) > 0 and on_point and not modifier

    np.testing.assert_array_equal(layer._drag_start, start_position)

    # Simulate drag start on a different position
    offset_position = [initial_position[0] + 20, initial_position[1] + 20]
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            position=offset_position,
            modifiers=modifier,
        )
    )
    mouse_move_callbacks(layer, event)

    # Initial mouse_move is already considered a move and not a press.
    # Therefore, the _drag_start value should be identical and the data or drag_box should reflect
    # the mouse position.
    np.testing.assert_array_equal(layer._drag_start, start_position)
    if is_point_move:
        if 0 in layer.selected_data:
            np.testing.assert_array_equal(
                layer.data[0].mean(axis=0),
                [offset_position[0], offset_position[1]],
            )
        else:
            assert False, 'Unreachable code'  # pragma: no cover
    else:
        np.testing.assert_array_equal(
            layer._drag_box, [initial_position, offset_position]
        )

    # Simulate drag start on new different position
    offset_position = zero_pos
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            position=offset_position,
            modifiers=modifier,
        )
    )
    mouse_move_callbacks(layer, event)

    # Initial mouse_move is already considered a move and not a press.
    # Therefore, the _drag_start value should be identical and the data or drag_box should reflect
    # the mouse position.
    np.testing.assert_array_equal(layer._drag_start, start_position)
    if is_point_move:
        if 0 in layer.selected_data:
            np.testing.assert_array_equal(
                layer.data[0].mean(axis=0),
                [offset_position[0], offset_position[1]],
            )
        else:
            assert False, 'Unreachable code'  # pragma: no cover
    else:
        np.testing.assert_array_equal(
            layer._drag_box, [initial_position, offset_position]
        )

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=True,
            modifiers=modifier,
            position=offset_position,
        )
    )
    mouse_release_callbacks(layer, event)

    if on_point and 0 in pre_selection and modifier:
        assert layer.selected_data == pre_selection - {0}
    elif on_point and 0 in pre_selection and not modifier:
        assert layer.selected_data == pre_selection
    elif on_point and 0 not in pre_selection and modifier:
        assert layer.selected_data == pre_selection | {0}
    elif on_point and 0 not in pre_selection and not modifier:
        assert layer.selected_data == {0}
    elif 0 in pre_selection and modifier:
        assert 0 not in layer.selected_data
        assert layer.selected_data == (set(range(n_points)) - pre_selection)
    elif 0 in pre_selection and not modifier:
        assert 0 in layer.selected_data
        assert layer.selected_data == set(range(n_points))
    elif 0 not in pre_selection and modifier:
        assert 0 in layer.selected_data
        assert layer.selected_data == (set(range(n_points)) - pre_selection)
    elif 0 not in pre_selection and not modifier:
        assert 0 in layer.selected_data
        assert layer.selected_data == set(range(n_points))
    else:
        assert False, 'Unreachable code'  # pragma: no cover
    assert layer._drag_box is None
    assert layer._drag_start is None
