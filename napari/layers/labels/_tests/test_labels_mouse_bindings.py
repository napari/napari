import collections

import numpy as np
import pytest

from napari.layers import Labels
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
        NamedTuple object with fields "type" and "is_dragging".
    """
    return collections.namedtuple(
        'Event', field_names=['type', 'is_dragging', 'position']
    )


@pytest.mark.parametrize(
    "brush_shape, expected_sum", [("circle", 244), ("square", 274)]
)
def test_paint(Event, brush_shape, expected_sum):
    """Test painting labels with circle/square brush."""
    data = np.ones((20, 20), dtype=np.int32)
    layer = Labels(data)
    layer.brush_size = 10
    assert layer.cursor_size == 10
    with pytest.warns(FutureWarning):
        layer.brush_shape = brush_shape

    layer.mode = 'paint'
    layer.selected_label = 3

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, position=(0, 0))
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag
    event = ReadOnlyWrapper(
        Event(type='mouse_move', is_dragging=True, position=(19, 19))
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=False, position=(19, 19))
    )
    mouse_release_callbacks(layer, event)

    # Painting goes from (0, 0) to (19, 19) with a brush size of 10, changing
    # all pixels along that path, but none outside it.
    assert np.unique(layer.data[:8, :8]) == 3
    assert np.unique(layer.data[-8:, -8:]) == 3
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1
    assert np.sum(layer.data == 3) == expected_sum


@pytest.mark.parametrize(
    "brush_shape, expected_sum", [("circle", 244), ("square", 274)]
)
def test_paint_scale(Event, brush_shape, expected_sum):
    """Test painting labels with circle/square brush when scaled."""
    data = np.ones((20, 20), dtype=np.int32)
    layer = Labels(data, scale=(2, 2))
    layer.brush_size = 10
    with pytest.warns(FutureWarning):
        layer.brush_shape = brush_shape

    layer.mode = 'paint'
    layer.selected_label = 3

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, position=(0, 0))
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag
    event = ReadOnlyWrapper(
        Event(type='mouse_move', is_dragging=True, position=(39, 39))
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=False, position=(39, 39))
    )
    mouse_release_callbacks(layer, event)

    # Painting goes from (0, 0) to (19, 19) with a brush size of 10, changing
    # all pixels along that path, but none outside it.
    assert np.unique(layer.data[:8, :8]) == 3
    assert np.unique(layer.data[-8:, -8:]) == 3
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1
    assert np.sum(layer.data == 3) == expected_sum


@pytest.mark.parametrize(
    "brush_shape, expected_sum", [("circle", 156), ("square", 126)]
)
def test_erase(Event, brush_shape, expected_sum):
    """Test erasing labels with different brush shapes."""
    data = np.ones((20, 20), dtype=np.int32)
    layer = Labels(data)
    with pytest.warns(FutureWarning):
        layer.brush_shape = brush_shape

    layer.brush_size = 10
    layer.mode = 'erase'

    with pytest.warns(FutureWarning):
        layer.brush_shape = brush_shape

    layer.selected_label = 3

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, position=(0, 0))
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag
    event = ReadOnlyWrapper(
        Event(type='mouse_move', is_dragging=True, position=(19, 19))
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(type='mouse_release', is_dragging=False, position=(19, 19))
    )
    mouse_release_callbacks(layer, event)

    # Painting goes from (0, 0) to (19, 19) with a brush size of 10, changing
    # all pixels along that path, but non outside it.
    assert np.unique(layer.data[:8, :8]) == 0
    assert np.unique(layer.data[-8:, -8:]) == 0
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1
    assert np.sum(layer.data == 1) == expected_sum


def test_pick(Event):
    """Test picking label."""
    data = np.ones((20, 20), dtype=np.int32)
    data[:5, :5] = 2
    data[-5:, -5:] = 3
    layer = Labels(data)
    assert layer.selected_label == 1

    layer.mode = 'pick'

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, position=(0, 0))
    )
    mouse_press_callbacks(layer, event)
    assert layer.selected_label == 2

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, position=(19, 19))
    )
    mouse_press_callbacks(layer, event)
    assert layer.selected_label == 3


def test_fill(Event):
    """Test filling label."""
    data = np.ones((20, 20), dtype=np.int32)
    data[:5, :5] = 2
    data[-5:, -5:] = 3
    layer = Labels(data)
    assert np.unique(layer.data[:5, :5]) == 2
    assert np.unique(layer.data[-5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1

    layer.mode = 'fill'
    layer.selected_label = 4

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, position=(0, 0))
    )
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1

    layer.selected_label = 5

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, position=(19, 19))
    )
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:]) == 5
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1


def test_fill_nD_plane(Event):
    """Test filling label nD plane."""
    data = np.ones((20, 20, 20), dtype=np.int32)
    data[:5, :5, :5] = 2
    data[0, 8:10, 8:10] = 2
    data[-5:, -5:, -5:] = 3
    layer = Labels(data)
    assert np.unique(layer.data[:5, :5, :5]) == 2
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 1
    assert np.unique(layer.data[-5:, :5, -5:]) == 1
    assert np.unique(layer.data[0, 8:10, 8:10]) == 2

    layer.mode = 'fill'
    layer.selected_label = 4

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, position=(0, 0, 0))
    )
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[0, :5, :5]) == 4
    assert np.unique(layer.data[1:5, :5, :5]) == 2
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 1
    assert np.unique(layer.data[-5:, :5, -5:]) == 1
    assert np.unique(layer.data[0, 8:10, 8:10]) == 2

    layer.selected_label = 5

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, position=(0, 19, 19))
    )
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[0, :5, :5]) == 4
    assert np.unique(layer.data[1:5, :5, :5]) == 2
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[1:5, -5:, -5:]) == 1
    assert np.unique(layer.data[-5:, :5, -5:]) == 1
    assert np.unique(layer.data[0, -5:, -5:]) == 5
    assert np.unique(layer.data[0, :5, -5:]) == 5
    assert np.unique(layer.data[0, 8:10, 8:10]) == 2


def test_fill_nD_all(Event):
    """Test filling label nD."""
    data = np.ones((20, 20, 20), dtype=np.int32)
    data[:5, :5, :5] = 2
    data[0, 8:10, 8:10] = 2
    data[-5:, -5:, -5:] = 3
    layer = Labels(data)
    assert np.unique(layer.data[:5, :5, :5]) == 2
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 1
    assert np.unique(layer.data[-5:, :5, -5:]) == 1
    assert np.unique(layer.data[0, 8:10, 8:10]) == 2

    layer.n_edit_dimensions = 3
    layer.mode = 'fill'
    layer.selected_label = 4

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, position=(0, 0, 0))
    )
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 1
    assert np.unique(layer.data[-5:, :5, -5:]) == 1
    assert np.unique(layer.data[0, 8:10, 8:10]) == 2

    layer.selected_label = 5

    # Simulate click
    event = ReadOnlyWrapper(
        Event(type='mouse_press', is_dragging=False, position=(0, 19, 19))
    )
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 5
    assert np.unique(layer.data[-5:, :5, -5:]) == 5
    assert np.unique(layer.data[0, 8:10, 8:10]) == 2
