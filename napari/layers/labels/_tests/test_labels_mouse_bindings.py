import numpy as np
import collections
from napari.layers import Labels
from napari.utils.interactions import (
    ReadOnlyWrapper,
    mouse_press_callbacks,
    mouse_move_callbacks,
    mouse_release_callbacks,
)


def test_mouse_move():
    """Test painting labels with different brush sizes."""
    np.random.seed(0)
    data = np.random.randint(20, size=(20, 20))
    layer = Labels(data)
    layer.brush_size = 10
    layer.mode = 'paint'
    layer.selected_label = 3
    layer._last_cursor_coord = (0, 0)
    layer.coordinates = (19, 19)

    Event = collections.namedtuple('Event', 'type is_dragging')

    # Simulate click
    event = ReadOnlyWrapper(Event(type='mouse_press', is_dragging=False))
    mouse_press_callbacks(layer, event)

    # Simulate drag
    event = ReadOnlyWrapper(Event(type='mouse_move', is_dragging=True))
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(Event(type='mouse_release', is_dragging=False))
    mouse_release_callbacks(layer, event)

    assert np.unique(layer.data[:5, :5]) == 3
    assert np.unique(layer.data[-5:, -5:]) == 3
