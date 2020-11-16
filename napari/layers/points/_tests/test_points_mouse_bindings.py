import collections

import numpy as np
import pytest

from napari.layers import Points
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
        'Event', field_names=['type', 'is_dragging', 'modifiers']
    )


@pytest.fixture
def create_known_points_layer():
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


def test_not_adding_or_selecting_point(create_known_points_layer, Event):
    """Don't add or select a point by clicking on one in pan_zoom mode."""
    layer, n_points, _ = create_known_points_layer
    layer.mode = 'pan_zoom'

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, modifiers=[])
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=False, modifiers=[])
    )
    mouse_release_callbacks(layer, event)

    # Check no new point added and non selected
    assert len(layer.data) == n_points
    assert len(layer.selected_data) == 0


def test_add_point(create_known_points_layer, Event):
    """Add point by clicking in add mode."""
    layer, n_points, known_non_point = create_known_points_layer

    # Add point at location where non exists
    layer.mode = 'add'
    layer.position = known_non_point

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, modifiers=[])
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=False, modifiers=[])
    )
    mouse_release_callbacks(layer, event)

    # Check new point added at coordinates location
    assert len(layer.data) == n_points + 1
    np.testing.assert_allclose(layer.data[-1], known_non_point)


def test_select_point(create_known_points_layer, Event):
    """Select a point by clicking on one in select mode."""
    layer, n_points, _ = create_known_points_layer

    layer.mode = 'select'
    layer.position = tuple(layer.data[0])

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, modifiers=[])
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=False, modifiers=[])
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 1
    assert 0 in layer.selected_data


def test_after_in_add_mode_point(create_known_points_layer, Event):
    """Don't add or select a point by clicking on one in pan_zoom mode."""
    layer, n_points, _ = create_known_points_layer

    layer.mode = 'add'
    layer.mode = 'pan_zoom'
    layer.position = tuple(layer.data[0])

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, modifiers=[])
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=False, modifiers=[])
    )
    mouse_release_callbacks(layer, event)

    # Check no new point added and non selected
    assert len(layer.data) == n_points
    assert len(layer.selected_data) == 0


def test_after_in_select_mode_point(create_known_points_layer, Event):
    """Don't add or select a point by clicking on one in pan_zoom mode."""
    layer, n_points, _ = create_known_points_layer

    layer.mode = 'select'
    layer.mode = 'pan_zoom'
    layer.position = tuple(layer.data[0])

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, modifiers=[])
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=False, modifiers=[])
    )
    mouse_release_callbacks(layer, event)

    # Check no new point added and non selected
    assert len(layer.data) == n_points
    assert len(layer.selected_data) == 0


def test_unselect_select_point(create_known_points_layer, Event):
    """Select a point by clicking on one in select mode."""
    layer, n_points, _ = create_known_points_layer

    layer.mode = 'select'
    layer.position = tuple(layer.data[0])
    layer.selected_data = {2, 3}

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, modifiers=[])
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=False, modifiers=[])
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 1
    assert 0 in layer.selected_data


def test_add_select_point(create_known_points_layer, Event):
    """Add to a selection of points point by shift-clicking on one."""
    layer, n_points, _ = create_known_points_layer

    layer.mode = 'select'
    layer.position = tuple(layer.data[0])
    layer.selected_data = {2, 3}

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, modifiers=['Shift'])
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=False, modifiers=['Shift'])
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 3
    assert layer.selected_data == {2, 3, 0}


def test_remove_select_point(create_known_points_layer, Event):
    """Remove from a selection of points point by shift-clicking on one."""
    layer, n_points, _ = create_known_points_layer

    layer.mode = 'select'
    layer.position = tuple(layer.data[0])
    layer.selected_data = {0, 2, 3}

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, modifiers=['Shift'])
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=False, modifiers=['Shift'])
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 2
    assert layer.selected_data == {2, 3}


def test_not_selecting_point(create_known_points_layer, Event):
    """Don't select a point by not clicking on one in select mode."""
    layer, n_points, known_non_point = create_known_points_layer

    layer.mode = 'select'
    layer.position = known_non_point

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, modifiers=[])
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=False, modifiers=[])
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 0


def test_unselecting_points(create_known_points_layer, Event):
    """Unselect points by not clicking on one in select mode."""
    layer, n_points, known_non_point = create_known_points_layer

    layer.mode = 'select'
    layer.position = known_non_point
    layer.selected_data = {2, 3}
    assert len(layer.selected_data) == 2

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, modifiers=[])
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=False, modifiers=[])
    )
    mouse_release_callbacks(layer, event)

    # Check clicked point selected
    assert len(layer.selected_data) == 0


def test_selecting_all_points_with_drag(create_known_points_layer, Event):
    """Select all points when drag box includes all of them."""
    layer, n_points, known_non_point = create_known_points_layer

    layer.mode = 'select'
    layer.position = known_non_point

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, modifiers=[])
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag start
    event = ReadOnlyWrapper(
        Event(type='mouse_move', is_dragging=True, modifiers=[])
    )
    mouse_move_callbacks(layer, event)

    layer.position = [0, 0]
    # Simulate drag end
    event = ReadOnlyWrapper(
        Event(type='mouse_move', is_dragging=True, modifiers=[])
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=True, modifiers=[])
    )
    mouse_release_callbacks(layer, event)

    # Check all points selected as drag box contains them
    assert len(layer.selected_data) == n_points


def test_selecting_no_points_with_drag(create_known_points_layer, Event):
    """Select all points when drag box includes all of them."""
    layer, n_points, known_non_point = create_known_points_layer

    layer.mode = 'select'
    layer.position = known_non_point

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, modifiers=[])
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag start
    event = ReadOnlyWrapper(
        Event(type='mouse_move', is_dragging=True, modifiers=[])
    )
    mouse_move_callbacks(layer, event)

    layer.position = [50, 60]
    # Simulate drag end
    event = ReadOnlyWrapper(
        Event(type='mouse_move', is_dragging=True, modifiers=[])
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=True, modifiers=[])
    )
    mouse_release_callbacks(layer, event)

    # Check no points selected as drag box doesn't contain them
    assert len(layer.selected_data) == 0
