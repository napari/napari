from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest

from napari.layers import Points
from napari.utils._proxies import ReadOnlyWrapper
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)


@dataclass
class Event:
    """Create a subclass for simulating vispy mouse events."""

    type: str
    is_dragging: bool = False
    modifiers: List[str] = field(default_factory=list)
    position: Union[Tuple[int, int], Tuple[int, int, int]] = (
        0,
        0,
    )  # world coords
    pos: np.ndarray = np.zeros(2)  # canvas coords
    view_direction: Optional[List[float]] = None
    up_direction: Optional[List[float]] = None
    dims_displayed: List[int] = field(default_factory=lambda: [0, 1])


@pytest.fixture
def create_known_points_layer_2d():
    """Create points layer with known coordinates

    Returns
    -------
    layer : napari.layers.Points
        Points layer.
    n_points : int
        Number of points in the points layer
    known_non_point : list
        Data coordinates that are known to contain no points. Useful during
        testing when needing to guarantee no point is clicked on.
    """
    data = [[1, 3], [8, 4], [10, 10], [15, 4]]
    known_non_point = [20, 30]
    n_points = len(data)

    layer = Points(data, size=1)
    assert np.all(layer.data == data)
    assert layer.ndim == 2
    assert len(layer.data) == n_points
    assert len(layer.selected_data) == 0

    return layer, n_points, known_non_point


@pytest.fixture
def create_known_points_layer_3d():
    """Create points layer with known coordinates

    Returns
    -------
    layer : napari.layers.Points
        Points layer.
    n_points : int
        Number of points in the points layer
    known_non_point : list
        Data coordinates that are known to contain no points. Useful during
        testing when needing to guarantee no point is clicked on.
    """
    data = [[1, 2, 3], [8, 6, 4], [10, 5, 10], [15, 8, 4]]
    known_non_point = [4, 5, 6]
    n_points = len(data)

    layer = Points(data, size=1)
    # extra variables usually set when layer is added to viewer must be declared
    # for certain 3D related methods.
    # e.g. Points._display_bounding_box_augmented, Points.get_ray_intersections
    layer._indices_view = np.array([0, 1, 2, 3])
    layer._ndisplay = 3

    assert np.all(layer.data == data)
    assert layer.ndim == 3
    assert len(layer._dims_displayed) == 3
    assert len(layer.data) == n_points
    assert len(layer._view_size) == n_points
    assert len(layer.selected_data) == 0

    return layer, n_points, known_non_point


def test_not_adding_or_selecting_point(create_known_points_layer_2d):
    """Don't add or select a point by clicking on one in pan_zoom mode."""
    layer, n_points, _ = create_known_points_layer_2d
    layer.mode = 'pan_zoom'

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press'))
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(Event(type='mouse_release'))
    mouse_release_callbacks(layer, event)

    # Check no new point added and non selected
    assert len(layer.data) == n_points
    assert len(layer.selected_data) == 0


def test_add_point(create_known_points_layer_2d):
    """Add point by clicking in add mode."""
    layer, n_points, known_non_point = create_known_points_layer_2d

    # Add point at location where non exists
    layer.mode = 'add'

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', position=known_non_point)
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', position=known_non_point)
    )
    mouse_release_callbacks(layer, event)

    # Check new point added at coordinates location
    assert len(layer.data) == n_points + 1
    np.testing.assert_allclose(layer.data[-1], known_non_point)


def test_add_point_3d(create_known_points_layer_3d):
    """Add a point by clicking in 3D mode."""
    layer, n_points, known_not_point = create_known_points_layer_3d

    layer.mode = 'add'

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            position=known_not_point,
            view_direction=[1, 0, 0],
            dims_displayed=[0, 1, 2],
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', position=known_not_point)
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.data) == (n_points + 1)
    np.testing.assert_array_equal(layer.data[-1], known_not_point)


def test_drag_in_add_mode(create_known_points_layer_2d):
    """Drag in add mode and make sure no point is added."""
    layer, n_points, known_non_point = create_known_points_layer_2d

    # Add point at location where non exists
    layer.mode = 'add'
    layer.interactive = True

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', position=known_non_point)
    )
    mouse_press_callbacks(layer, event)

    known_non_point_end = [40, 60]

    # Simulate drag end
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move', is_dragging=True, position=known_non_point_end
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            position=known_non_point_end,
            pos=np.array([4, 4]),
        )
    )
    mouse_release_callbacks(layer, event)

    # Check that no new point has been added
    assert len(layer.data) == n_points


def test_select_point(create_known_points_layer_2d):
    """Select a point by clicking on one in select mode."""
    layer, n_points, _ = create_known_points_layer_2d

    layer.mode = 'select'
    position = tuple(layer.data[0])

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', position=position))
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(Event(type='mouse_release', position=position))
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 1
    assert 0 in layer.selected_data


def test_select_point_3d(create_known_points_layer_3d):
    """Select a point by clicking on one in select mode in 3D mode."""
    layer, n_points, _ = create_known_points_layer_3d

    layer.mode = 'select'
    position = tuple(layer.data[1])

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            position=position,
            view_direction=[1, 0, 0],
            dims_displayed=[0, 1, 2],
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(Event(type='mouse_release', position=position))
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 1
    assert 1 in layer.selected_data


def test_unselect_by_click_point_3d(create_known_points_layer_3d):
    """Select unselecting point by shift clicking on it again in 3D mode."""
    layer, n_points, _ = create_known_points_layer_3d

    layer.mode = 'select'
    position = tuple(layer.data[1])

    layer.selected_data = {0, 1}

    # Simulate shift+click on point 1
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            position=position,
            modifiers=['Shift'],
            view_direction=[1, 0, 0],
            dims_displayed=[0, 1, 2],
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', modifiers=['Shift'], position=position)
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert layer.selected_data == {0}


def test_select_by_shift_click_3d(create_known_points_layer_3d):
    """Select selecting point by shift clicking on an additional point in 3D"""
    layer, n_points, _ = create_known_points_layer_3d

    layer.mode = 'select'
    position = tuple(layer.data[1])

    layer.selected_data = {0}

    # Simulate shift+click on point 1
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            position=position,
            modifiers=['Shift'],
            view_direction=[1, 0, 0],
            dims_displayed=[0, 1, 2],
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', modifiers=['Shift'], position=position)
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert layer.selected_data == {0, 1}


def test_unselect_by_click_empty_3d(create_known_points_layer_3d):
    """Select unselecting point by clicking in empty space"""
    layer, n_points, known_not_point = create_known_points_layer_3d

    layer.mode = 'select'

    layer.selected_data = {0, 1}

    # Simulate click on point
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            position=known_not_point,
            view_direction=[1, 0, 0],
            dims_displayed=[0, 1, 2],
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', position=known_not_point)
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 0


def test_after_in_add_mode_point(create_known_points_layer_2d):
    """Don't add or select a point by clicking on one in pan_zoom mode."""
    layer, n_points, _ = create_known_points_layer_2d

    layer.mode = 'add'
    layer.mode = 'pan_zoom'
    position = tuple(layer.data[0])

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', position=position))
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(Event(type='mouse_release', position=position))
    mouse_release_callbacks(layer, event)

    # Check no new point added and non selected
    assert len(layer.data) == n_points
    assert len(layer.selected_data) == 0


def test_after_in_select_mode_point(create_known_points_layer_2d):
    """Don't add or select a point by clicking on one in pan_zoom mode."""
    layer, n_points, _ = create_known_points_layer_2d

    layer.mode = 'select'
    layer.mode = 'pan_zoom'
    position = tuple(layer.data[0])

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', position=position))
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(Event(type='mouse_release', position=position))
    mouse_release_callbacks(layer, event)

    # Check no new point added and non selected
    assert len(layer.data) == n_points
    assert len(layer.selected_data) == 0


def test_unselect_select_point(create_known_points_layer_2d):
    """Select a point by clicking on one in select mode."""
    layer, n_points, _ = create_known_points_layer_2d

    layer.mode = 'select'
    position = tuple(layer.data[0])
    layer.selected_data = {2, 3}

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', position=position))
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(Event(type='mouse_release', position=position))
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 1
    assert 0 in layer.selected_data


def test_add_select_point(create_known_points_layer_2d):
    """Add to a selection of points point by shift-clicking on one."""
    layer, n_points, _ = create_known_points_layer_2d

    layer.mode = 'select'
    position = tuple(layer.data[0])
    layer.selected_data = {2, 3}

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', modifiers=['Shift'], position=position)
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', modifiers=['Shift'], position=position)
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 3
    assert layer.selected_data == {2, 3, 0}


def test_remove_select_point(create_known_points_layer_2d):
    """Remove from a selection of points point by shift-clicking on one."""
    layer, n_points, _ = create_known_points_layer_2d

    layer.mode = 'select'
    position = tuple(layer.data[0])
    layer.selected_data = {0, 2, 3}

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', modifiers=['Shift'], position=position)
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', modifiers=['Shift'], position=position)
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 2
    assert layer.selected_data == {2, 3}


def test_not_selecting_point(create_known_points_layer_2d):
    """Don't select a point by not clicking on one in select mode."""
    layer, n_points, known_non_point = create_known_points_layer_2d

    layer.mode = 'select'

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', position=known_non_point)
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', position=known_non_point)
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 0


def test_unselecting_points(create_known_points_layer_2d):
    """Unselect points by not clicking on one in select mode."""
    layer, n_points, known_non_point = create_known_points_layer_2d

    layer.mode = 'select'
    layer.selected_data = {2, 3}
    assert len(layer.selected_data) == 2

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', position=known_non_point)
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', position=known_non_point)
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 0


def test_selecting_all_points_with_drag_2d(create_known_points_layer_2d):
    """Select all points when drag box includes all of them."""
    layer, n_points, known_non_point = create_known_points_layer_2d

    layer.mode = 'select'

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', position=known_non_point)
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag start
    event = ReadOnlyWrapper(
        Event(type='mouse_move', is_dragging=True, position=known_non_point)
    )
    mouse_move_callbacks(layer, event)

    # Simulate drag end
    event = ReadOnlyWrapper(Event(type='mouse_move', is_dragging=True))
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(Event(type='mouse_release', is_dragging=True))
    mouse_release_callbacks(layer, event)

    # Check all points selected as drag box contains them
    assert len(layer.selected_data) == n_points


def test_selecting_no_points_with_drag_2d(create_known_points_layer_2d):
    """Select no points when drag box outside of all of them."""
    layer, n_points, known_non_point = create_known_points_layer_2d

    layer.mode = 'select'

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', position=known_non_point)
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag start
    event = ReadOnlyWrapper(
        Event(type='mouse_move', is_dragging=True, position=known_non_point)
    )
    mouse_move_callbacks(layer, event)

    # Simulate drag end
    event = ReadOnlyWrapper(
        Event(type='mouse_move', is_dragging=True, position=(50, 60))
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=True, position=(50, 60))
    )
    mouse_release_callbacks(layer, event)

    # Check no points selected as drag box doesn't contain them
    assert len(layer.selected_data) == 0


def test_selecting_points_with_drag_3d(create_known_points_layer_3d):
    """Select all points when drag box includes all of them."""
    layer, n_points, known_non_point = create_known_points_layer_3d

    layer.mode = 'select'

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            position=(5, 0, 0),
            view_direction=[1, 0, 0],
            up_direction=[0, 1, 0],
            dims_displayed=[0, 1, 2],
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag start
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            position=(5, 0, 0),
            view_direction=[1, 0, 0],
            up_direction=[0, 1, 0],
            dims_displayed=[0, 1, 2],
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate drag end
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            position=(5, 6, 6),
            view_direction=[1, 0, 0],
            up_direction=[0, 1, 0],
            dims_displayed=[0, 1, 2],
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=True,
            position=(5, 6, 6),
            view_direction=[1, 0, 0],
            up_direction=[0, 1, 0],
            dims_displayed=[0, 1, 2],
        )
    )
    mouse_release_callbacks(layer, event)

    # Check all points selected as drag box contains them
    assert layer.selected_data == {0, 1}


def test_selecting_no_points_with_drag_3d(create_known_points_layer_3d):
    """Select no points when drag box outside of all of them."""
    layer, n_points, known_non_point = create_known_points_layer_3d

    layer.mode = 'select'

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            position=(5, 15, 15),
            view_direction=[1, 0, 0],
            up_direction=[0, 1, 0],
            dims_displayed=[0, 1, 2],
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag start
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            position=(5, 15, 15),
            view_direction=[1, 0, 0],
            up_direction=[0, 1, 0],
            dims_displayed=[0, 1, 2],
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate drag end
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            position=(5, 20, 20),
            view_direction=[1, 0, 0],
            up_direction=[0, 1, 0],
            dims_displayed=[0, 1, 2],
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=True,
            position=(5, 20, 20),
            view_direction=[1, 0, 0],
            up_direction=[0, 1, 0],
            dims_displayed=[0, 1, 2],
        )
    )
    mouse_release_callbacks(layer, event)

    # Check all points selected as drag box contains them
    assert len(layer.selected_data) == 0


@pytest.mark.parametrize(
    'pre_selection,on_point,modifier',
    [
        (set(), True, []),
        ({0}, True, []),
        ({0, 1, 2}, True, []),
        ({1, 2}, True, []),
        (set(), True, ['Shift']),
        ({0}, True, ['Shift']),
        ({0, 1, 2}, True, ['Shift']),
        ({1, 2}, True, ['Shift']),
        (set(), False, []),
        ({0}, False, []),
        ({0, 1, 2}, False, []),
        ({1, 2}, False, []),
        (set(), False, ['Shift']),
        ({0}, False, ['Shift']),
        ({0, 1, 2}, False, ['Shift']),
        ({1, 2}, False, ['Shift']),
    ],
)
def test_drag_start_selection(
    create_known_points_layer_2d, pre_selection, on_point, modifier
):
    """Check layer drag start and drag box behave as expected."""
    layer, n_points, known_non_point = create_known_points_layer_2d
    layer.mode = 'select'
    layer.selected_data = pre_selection

    if on_point:
        initial_position = tuple(layer.data[0])
    else:
        initial_position = tuple(known_non_point)
    zero_pos = [0, 0]
    initial_position_1 = tuple(layer.data[1])
    diff_data_1 = [
        layer.data[1, 0] - layer.data[0, 0],
        layer.data[1, 1] - layer.data[0, 1],
    ]

    assert layer._drag_start is None
    assert layer._drag_box is None

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press', position=initial_position, modifiers=modifier
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
        center = layer.data[list(layer.selected_data), :].mean(axis=0)
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
        if 1 in layer.selected_data and 0 in layer.selected_data:
            np.testing.assert_array_equal(
                layer.data[1],
                [
                    offset_position[0] + diff_data_1[0],
                    offset_position[1] + diff_data_1[1],
                ],
            )
        elif 1 not in layer.selected_data:
            np.testing.assert_array_equal(layer.data[1], initial_position_1)

        if 0 in layer.selected_data:
            np.testing.assert_array_equal(
                layer.data[0], [offset_position[0], offset_position[1]]
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
        if 1 in layer.selected_data and 0 in layer.selected_data:
            np.testing.assert_array_equal(
                layer.data[1],
                [
                    offset_position[0] + diff_data_1[0],
                    offset_position[1] + diff_data_1[1],
                ],
            )
        elif 1 not in layer.selected_data:
            np.testing.assert_array_equal(layer.data[1], initial_position_1)

        if 0 in layer.selected_data:
            np.testing.assert_array_equal(
                layer.data[0], [offset_position[0], offset_position[1]]
            )
        else:
            assert False, 'Unreachable code'  # pragma: no cover
    else:
        np.testing.assert_array_equal(
            layer._drag_box, [initial_position, offset_position]
        )

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=True, modifiers=modifier)
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
