import numpy as np
from scipy import ndimage as ndi

from napari.layers import Labels
from napari.utils.interactions import (
    ReadOnlyWrapper,
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)


def test_paint(MouseEvent):
    """Test painting labels with circle brush."""
    data = np.ones((20, 20), dtype=np.int32)
    layer = Labels(data)
    layer.brush_size = 10
    assert layer.cursor_size == 10

    layer.mode = 'paint'
    layer.selected_label = 3

    # Simulate click
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(0, 0),
            view_direction=None,
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_move',
            is_dragging=True,
            position=(19, 19),
            view_direction=None,
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_release',
            is_dragging=False,
            position=(19, 19),
            view_direction=None,
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
    )
    mouse_release_callbacks(layer, event)

    # Painting goes from (0, 0) to (19, 19) with a brush size of 10, changing
    # all pixels along that path, but none outside it.
    assert np.unique(layer.data[:8, :8]) == 3
    assert np.unique(layer.data[-8:, -8:]) == 3
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1
    assert np.sum(layer.data == 3) == 244


def test_paint_scale(MouseEvent):
    """Test painting labels with circle brush when scaled."""
    data = np.ones((20, 20), dtype=np.int32)
    layer = Labels(data, scale=(2, 2))
    layer.brush_size = 10

    layer.mode = 'paint'
    layer.selected_label = 3

    # Simulate click
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(0, 0),
            view_direction=None,
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_move',
            is_dragging=True,
            position=(39, 39),
            view_direction=None,
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_release',
            is_dragging=False,
            position=(39, 39),
            view_direction=None,
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
    )
    mouse_release_callbacks(layer, event)

    # Painting goes from (0, 0) to (19, 19) with a brush size of 10, changing
    # all pixels along that path, but none outside it.
    assert np.unique(layer.data[:8, :8]) == 3
    assert np.unique(layer.data[-8:, -8:]) == 3
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1
    assert np.sum(layer.data == 3) == 244


def test_erase(MouseEvent):
    """Test erasing labels with different brush shapes."""
    data = np.ones((20, 20), dtype=np.int32)
    layer = Labels(data)
    layer.brush_size = 10
    layer.mode = 'erase'

    layer.selected_label = 3

    # Simulate click
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(0, 0),
            view_direction=None,
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_move',
            is_dragging=True,
            position=(19, 19),
            view_direction=None,
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_release',
            is_dragging=False,
            position=(19, 19),
            view_direction=None,
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
    )
    mouse_release_callbacks(layer, event)

    # Painting goes from (0, 0) to (19, 19) with a brush size of 10, changing
    # all pixels along that path, but non outside it.
    assert np.unique(layer.data[:8, :8]) == 0
    assert np.unique(layer.data[-8:, -8:]) == 0
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1
    assert np.sum(layer.data == 1) == 156


def test_pick(MouseEvent):
    """Test picking label."""
    data = np.ones((20, 20), dtype=np.int32)
    data[:5, :5] = 2
    data[-5:, -5:] = 3
    layer = Labels(data)
    assert layer.selected_label == 1

    layer.mode = 'pick'

    # Simulate click
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(0, 0),
            view_direction=None,
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
    )
    mouse_press_callbacks(layer, event)
    assert layer.selected_label == 2

    # Simulate click
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(19, 19),
            view_direction=None,
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
    )
    mouse_press_callbacks(layer, event)
    assert layer.selected_label == 3


def test_fill(MouseEvent):
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
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(0, 0),
            view_direction=None,
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
    )
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1

    layer.selected_label = 5

    # Simulate click
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(19, 19),
            view_direction=None,
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
    )
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:]) == 5
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1


def test_fill_nD_plane(MouseEvent):
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
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(0, 0, 0),
            view_direction=(1, 0, 0),
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
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
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(0, 19, 19),
            view_direction=(1, 0, 0),
            dims_displayed=(0, 1),
            dims_point=(0, 0, 0),
        )
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


def test_fill_nD_all(MouseEvent):
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
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(0, 0, 0),
            view_direction=(1, 0, 0),
            dims_displayed=(0, 1),
            dims_point=(0, 0),
        )
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
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(0, 19, 19),
            view_direction=(1, 0, 0),
            dims_displayed=(0, 1),
            dims_point=(0, 0, 0),
        )
    )
    mouse_press_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 5
    assert np.unique(layer.data[-5:, :5, -5:]) == 5
    assert np.unique(layer.data[0, 8:10, 8:10]) == 2


def test_paint_3d(MouseEvent):
    """Test filling label nD."""
    data = np.zeros((21, 21, 21), dtype=np.int32)
    data[10, 10, 10] = 1
    layer = Labels(data)
    layer._slice_dims(point=(0, 0, 0), ndisplay=3)

    layer.n_edit_dimensions = 3
    layer.mode = 'paint'
    layer.selected_label = 4
    layer.brush_size = 3

    # Simulate click
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(0.1, 0, 0),
            view_direction=np.full(3, np.sqrt(3)),
            dims_displayed=(0, 1, 2),
            dims_point=(0, 0, 0),
        )
    )
    mouse_press_callbacks(layer, event)
    np.testing.assert_array_equal(np.unique(layer.data), [0, 4])
    num_filled = np.bincount(layer.data.ravel())[4]
    assert num_filled > 1

    layer.mode = 'erase'

    # Simulate click
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(0, 10, 10),
            view_direction=(1, 0, 0),
            dims_displayed=(0, 1, 2),
            dims_point=(0, 0, 0),
        )
    )
    mouse_press_callbacks(layer, event)

    new_num_filled = np.bincount(layer.data.ravel())[4]
    assert new_num_filled < num_filled


def test_erase_3d_undo(MouseEvent):
    """Test erasing labels in 3D then undoing the erase.

    Specifically, this test checks that undo is correctly filled even
    when a click and drag starts outside of the data volume.
    """
    data = np.zeros((20, 20, 20), dtype=np.int32)
    data[10, :, :] = 1
    layer = Labels(data)
    layer.brush_size = 5
    layer.mode = 'erase'
    layer._slice_dims(point=(0, 0, 0), ndisplay=3)
    layer.n_edit_dimensions = 3

    # Simulate click
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(-1, -1, -1),
            view_direction=(1, 0, 0),
            dims_displayed=(0, 1, 2),
            dims_point=(0, 0, 0),
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag. Note: we need to include top left and bottom right in the
    # drag or there are no coordinates to interpolate
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_move',
            is_dragging=True,
            position=(-1, 0.1, 0.1),
            view_direction=(1, 0, 0),
            dims_displayed=(0, 1, 2),
            dims_point=(0, 0, 0),
        )
    )
    mouse_move_callbacks(layer, event)
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_move',
            is_dragging=True,
            position=(-1, 18.9, 18.9),
            view_direction=(1, 0, 0),
            dims_displayed=(0, 1, 2),
            dims_point=(0, 0, 0),
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_release',
            is_dragging=False,
            position=(-1, 21, 21),
            view_direction=(1, 0, 0),
            dims_displayed=(0, 1, 2),
            dims_point=(0, 0, 0),
        )
    )
    mouse_release_callbacks(layer, event)

    # Erasing goes from (-1, -1, -1) to (-1, 21, 21), should split the labels
    # into two sections. Undoing should work and reunite the labels to one
    # square
    assert ndi.label(layer.data)[1] == 2
    layer.undo()
    assert ndi.label(layer.data)[1] == 1


def test_erase_3d_undo_empty(MouseEvent):
    """Nothing should be added to undo queue when clicks fall outside data."""
    data = np.zeros((20, 20, 20), dtype=np.int32)
    data[10, :, :] = 1
    layer = Labels(data)
    layer.brush_size = 5
    layer.mode = 'erase'
    layer._slice_dims(point=(0, 0, 0), ndisplay=3)
    layer.n_edit_dimensions = 3

    # Simulate click, outside data
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_press',
            is_dragging=False,
            position=(-1, -1, -1),
            view_direction=(1, 0, 0),
            dims_displayed=(0, 1, 2),
            dims_point=(0, 0, 0),
        )
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        MouseEvent(
            type='mouse_release',
            is_dragging=False,
            position=(-1, -1, -1),
            view_direction=(1, 0, 0),
            dims_displayed=(0, 1, 2),
            dims_point=(0, 0, 0),
        )
    )
    mouse_release_callbacks(layer, event)

    # Undo queue should be empty
    assert len(layer._undo_history) == 0
