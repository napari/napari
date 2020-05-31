import pytest
import numpy as np
import collections
from napari.layers import Labels
from napari.utils.interactions import (
    ReadOnlyWrapper,
    mouse_press_callbacks,
    mouse_move_callbacks,
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
    return collections.namedtuple('Event', field_names=['type', 'is_dragging'])


def test_paint(Event):
    """Test painting labels with different brush sizes."""
    data = np.ones((20, 20))
    layer = Labels(data)
    layer.brush_size = 10
    layer.mode = 'paint'
    layer.selected_label = 3
    layer.position = (0, 0)

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
    mouse_press_callbacks(layer, event)

    layer.position = (19, 19)

    # Simulate drag
    event = ReadOnlyWrapper(Event(type='mouse_move', is_dragging=True))
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(Event(type='mouse_release', is_dragging=False))
    mouse_release_callbacks(layer, event)

    # Painting goes from (0, 0) to (19, 19) with a brush size of 10, changing
    # all pixels along that path, but non outside it.
    assert np.unique(layer.data[:5, :5]) == 3
    assert np.unique(layer.data[-5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1


def test_erase(Event):
    """Test erasing labels with different brush sizes."""
    data = np.ones((20, 20))
    layer = Labels(data)
    layer.brush_size = 10
    layer.mode = 'erase'
    layer.selected_label = 3
    layer.position = (0, 0)

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
    mouse_press_callbacks(layer, event)

    layer.position = (19, 19)

    # Simulate drag
    event = ReadOnlyWrapper(Event(type='mouse_move', is_dragging=True))
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(Event(type='mouse_release', is_dragging=False))
    mouse_release_callbacks(layer, event)

    # Painting goes from (0, 0) to (19, 19) with a brush size of 10, changing
    # all pixels along that path, but non outside it.
    assert np.unique(layer.data[:5, :5]) == 0
    assert np.unique(layer.data[-5:, -5:]) == 0
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1


def test_pick(Event):
    """Test picking label."""
    data = np.ones((20, 20))
    data[:5, :5] = 2
    data[-5:, -5:] = 3
    layer = Labels(data)
    assert layer.selected_label == 1

    layer.mode = 'pick'
    layer.position = (0, 0)

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
    mouse_press_callbacks(layer, event)
    assert layer.selected_label == 2

    layer.position = (19, 19)

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
    mouse_press_callbacks(layer, event)
    assert layer.selected_label == 3


def test_fill(Event):
    """Test filling label."""
    data = np.ones((20, 20))
    data[:5, :5] = 2
    data[-5:, -5:] = 3
    layer = Labels(data)
    assert np.unique(layer.data[:5, :5]) == 2
    assert np.unique(layer.data[-5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1

    layer.mode = 'fill'
    layer.position = (0, 0)
    layer.selected_label = 4

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1

    layer.position = (19, 19)
    layer.selected_label = 5

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:]) == 5
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1


def test_fill_nD_plane(Event):
    """Test filling label nD plane."""
    data = np.ones((20, 20, 20))
    data[:5, :5, :5] = 2
    data[-5:, -5:, -5:] = 3
    layer = Labels(data)
    assert np.unique(layer.data[:5, :5, :5]) == 2
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 1
    assert np.unique(layer.data[-5:, :5, -5:]) == 1

    layer.mode = 'fill'
    layer.position = (0, 0)
    layer.selected_label = 4

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[0, :5, :5]) == 4
    assert np.unique(layer.data[1:5, :5, :5]) == 2
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 1
    assert np.unique(layer.data[-5:, :5, -5:]) == 1

    layer.position = (19, 19)
    layer.selected_label = 5

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[0, :5, :5]) == 4
    assert np.unique(layer.data[1:5, :5, :5]) == 2
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[1:5, -5:, -5:]) == 1
    assert np.unique(layer.data[-5:, :5, -5:]) == 1
    assert np.unique(layer.data[0, -5:, -5:]) == 5
    assert np.unique(layer.data[0, :5, -5:]) == 5


def test_fill_nD_all(Event):
    """Test filling label nD."""
    data = np.ones((20, 20, 20))
    data[:5, :5, :5] = 2
    data[-5:, -5:, -5:] = 3
    layer = Labels(data)
    assert np.unique(layer.data[:5, :5, :5]) == 2
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 1
    assert np.unique(layer.data[-5:, :5, -5:]) == 1
    layer.n_dimensional = True
    layer.mode = 'fill'
    layer.position = (0, 0)
    layer.selected_label = 4

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 1
    assert np.unique(layer.data[-5:, :5, -5:]) == 1

    layer.position = (19, 19)
    layer.selected_label = 5

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 5
    assert np.unique(layer.data[-5:, :5, -5:]) == 5
