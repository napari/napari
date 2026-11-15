import numpy as np
import pytest
from scipy import ndimage as ndi

from napari.components.dims import Dims
from napari.layers import Labels
from napari.utils._test_utils import read_only_mouse_event
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)


def test_paint():
    """Test painting labels with circle brush."""
    data = np.ones((20, 20), dtype=np.int32)
    layer = Labels(data)
    layer.brush_size = 10
    assert layer.cursor_size == 10

    layer.mode = 'paint'
    layer.selected_label = 3

    # Simulate click
    event = read_only_mouse_event(
        type='mouse_press',
        position=(0, 0),
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag
    event = read_only_mouse_event(
        type='mouse_move',
        is_dragging=True,
        position=(19, 19),
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = read_only_mouse_event(
        type='mouse_release',
        position=(19, 19),
    )
    mouse_release_callbacks(layer, event)

    # Painting goes from (0, 0) to (19, 19) with a brush size of 10, changing
    # all pixels along that path, but none outside it.
    assert np.unique(layer.data[:8, :8]) == 3
    assert np.unique(layer.data[-8:, -8:]) == 3
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1
    assert np.sum(layer.data == 3) == 244


def test_paint_scale():
    """Test painting labels with circle brush when scaled."""
    data = np.ones((20, 20), dtype=np.int32)
    layer = Labels(data, scale=(2, 2))
    layer.brush_size = 10

    layer.mode = 'paint'
    layer.selected_label = 3

    # Simulate click
    event = read_only_mouse_event(
        type='mouse_press',
        position=(0, 0),
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag
    event = read_only_mouse_event(
        type='mouse_move',
        is_dragging=True,
        position=(39, 39),
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = read_only_mouse_event(
        type='mouse_release',
        position=(39, 39),
    )
    mouse_release_callbacks(layer, event)

    # Painting goes from (0, 0) to (19, 19) with a brush size of 10, changing
    # all pixels along that path, but none outside it.
    assert np.unique(layer.data[:8, :8]) == 3
    assert np.unique(layer.data[-8:, -8:]) == 3
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1
    assert np.sum(layer.data == 3) == 244


def test_erase():
    """Test erasing labels with different brush shapes."""
    data = np.ones((20, 20), dtype=np.int32)
    layer = Labels(data)
    layer.brush_size = 10
    layer.mode = 'erase'

    layer.selected_label = 3

    # Simulate click
    event = read_only_mouse_event(
        type='mouse_press',
        position=(0, 0),
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag
    event = read_only_mouse_event(
        type='mouse_move',
        is_dragging=True,
        position=(19, 19),
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = read_only_mouse_event(
        type='mouse_release',
        position=(19, 19),
    )
    mouse_release_callbacks(layer, event)

    # Painting goes from (0, 0) to (19, 19) with a brush size of 10, changing
    # all pixels along that path, but non outside it.
    assert np.unique(layer.data[:8, :8]) == 0
    assert np.unique(layer.data[-8:, -8:]) == 0
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1
    assert np.sum(layer.data == 1) == 156


def test_pick():
    """Test picking label."""
    data = np.ones((20, 20), dtype=np.int32)
    data[:5, :5] = 2
    data[-5:, -5:] = 3
    layer = Labels(data)
    assert layer.selected_label == 1

    layer.mode = 'pick'

    # Simulate click
    event = read_only_mouse_event(
        type='mouse_press',
        position=(0, 0),
    )
    mouse_press_callbacks(layer, event)
    assert layer.selected_label == 2

    # Simulate click
    event = read_only_mouse_event(
        type='mouse_press',
        position=(19, 19),
    )
    mouse_press_callbacks(layer, event)
    assert layer.selected_label == 3


def test_fill():
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
    event = read_only_mouse_event(
        type='mouse_press',
        position=(0, 0),
    )
    mouse_press_callbacks(layer, event)
    mouse_release_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1

    layer.selected_label = 5

    # Simulate click
    event = read_only_mouse_event(
        type='mouse_press',
        position=(19, 19),
    )
    mouse_press_callbacks(layer, event)
    mouse_release_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:]) == 5
    assert np.unique(layer.data[:5, -5:]) == 1
    assert np.unique(layer.data[-5:, :5]) == 1


def test_fill_nD_plane():
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
    event = read_only_mouse_event(
        type='mouse_press',
        position=(0, 0, 0),
        view_direction=(1, 0, 0),
    )
    mouse_press_callbacks(layer, event)
    mouse_release_callbacks(layer, event)
    assert np.unique(layer.data[0, :5, :5]) == 4
    assert np.unique(layer.data[1:5, :5, :5]) == 2
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 1
    assert np.unique(layer.data[-5:, :5, -5:]) == 1
    assert np.unique(layer.data[0, 8:10, 8:10]) == 2

    layer.selected_label = 5

    # Simulate click
    event = read_only_mouse_event(
        type='mouse_press',
        position=(0, 19, 19),
        view_direction=(1, 0, 0),
        dims_point=(0, 0, 0),
    )
    mouse_press_callbacks(layer, event)
    mouse_release_callbacks(layer, event)
    assert np.unique(layer.data[0, :5, :5]) == 4
    assert np.unique(layer.data[1:5, :5, :5]) == 2
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[1:5, -5:, -5:]) == 1
    assert np.unique(layer.data[-5:, :5, -5:]) == 1
    assert np.unique(layer.data[0, -5:, -5:]) == 5
    assert np.unique(layer.data[0, :5, -5:]) == 5
    assert np.unique(layer.data[0, 8:10, 8:10]) == 2


def test_fill_nD_all():
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
    event = read_only_mouse_event(
        type='mouse_press',
        position=(0, 0, 0),
        view_direction=(1, 0, 0),
    )
    mouse_press_callbacks(layer, event)
    mouse_release_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 1
    assert np.unique(layer.data[-5:, :5, -5:]) == 1
    assert np.unique(layer.data[0, 8:10, 8:10]) == 2

    layer.selected_label = 5

    # Simulate click
    event = read_only_mouse_event(
        type='mouse_press',
        position=(0, 19, 19),
        view_direction=(1, 0, 0),
        dims_point=(0, 0, 0),
    )
    mouse_press_callbacks(layer, event)
    mouse_release_callbacks(layer, event)
    assert np.unique(layer.data[:5, :5, :5]) == 4
    assert np.unique(layer.data[-5:, -5:, -5:]) == 3
    assert np.unique(layer.data[:5, -5:, -5:]) == 5
    assert np.unique(layer.data[-5:, :5, -5:]) == 5
    assert np.unique(layer.data[0, 8:10, 8:10]) == 2


def test_paint_3d():
    """Test filling label nD."""
    data = np.zeros((21, 21, 21), dtype=np.int32)
    data[10, 10, 10] = 1
    layer = Labels(data)
    layer._slice_dims(Dims(ndim=3, ndisplay=3))

    layer.n_edit_dimensions = 3
    layer.mode = 'paint'
    layer.selected_label = 4
    layer.brush_size = 3

    # Simulate click
    event = read_only_mouse_event(
        type='mouse_press',
        position=(0.1, 0, 0),
        view_direction=np.full(3, np.sqrt(3)),
        dims_displayed=(0, 1, 2),
        dims_point=(0, 0, 0),
    )
    mouse_press_callbacks(layer, event)
    mouse_release_callbacks(layer, event)
    np.testing.assert_array_equal(np.unique(layer.data), [0, 4])
    num_filled = np.bincount(layer.data.ravel())[4]
    assert num_filled > 1

    layer.mode = 'erase'

    # Simulate click
    event = read_only_mouse_event(
        type='mouse_press',
        position=(0, 10, 10),
        view_direction=(1, 0, 0),
        dims_displayed=(0, 1, 2),
        dims_point=(0, 0, 0),
    )
    mouse_press_callbacks(layer, event)
    mouse_release_callbacks(layer, event)

    new_num_filled = np.bincount(layer.data.ravel())[4]
    assert new_num_filled < num_filled


def test_erase_3d_undo():
    """Test erasing labels in 3D then undoing the erase.

    Specifically, this test checks that undo is correctly filled even
    when a click and drag starts outside of the data volume.
    """
    data = np.zeros((20, 20, 20), dtype=np.int32)
    data[10, :, :] = 1
    layer = Labels(data)
    layer.brush_size = 5
    layer.mode = 'erase'
    layer._slice_dims(Dims(ndim=3, ndisplay=3))
    layer.n_edit_dimensions = 3

    # Simulate click
    event = read_only_mouse_event(
        type='mouse_press',
        position=(-1, -1, -1),
        view_direction=(1, 0, 0),
        dims_displayed=(0, 1, 2),
        dims_point=(0, 0, 0),
    )
    mouse_press_callbacks(layer, event)

    # Simulate drag. Note: we need to include top left and bottom right in the
    # drag or there are no coordinates to interpolate
    event = read_only_mouse_event(
        type='mouse_move',
        is_dragging=True,
        position=(-1, 0.1, 0.1),
        view_direction=(1, 0, 0),
        dims_displayed=(0, 1, 2),
        dims_point=(0, 0, 0),
    )
    mouse_move_callbacks(layer, event)
    event = read_only_mouse_event(
        type='mouse_move',
        is_dragging=True,
        position=(-1, 18.9, 18.9),
        view_direction=(1, 0, 0),
        dims_displayed=(0, 1, 2),
        dims_point=(0, 0, 0),
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = read_only_mouse_event(
        type='mouse_release',
        position=(-1, 21, 21),
        view_direction=(1, 0, 0),
        dims_displayed=(0, 1, 2),
        dims_point=(0, 0, 0),
    )
    mouse_release_callbacks(layer, event)

    # Erasing goes from (-1, -1, -1) to (-1, 21, 21), should split the labels
    # into two sections. Undoing should work and reunite the labels to one
    # square
    assert ndi.label(layer.data)[1] == 2
    layer.undo()
    assert ndi.label(layer.data)[1] == 1


def test_erase_3d_undo_empty():
    """Nothing should be added to undo queue when clicks fall outside data."""
    data = np.zeros((20, 20, 20), dtype=np.int32)
    data[10, :, :] = 1
    layer = Labels(data)
    layer.brush_size = 5
    layer.mode = 'erase'
    layer._slice_dims(Dims(ndim=3, ndisplay=3))
    layer.n_edit_dimensions = 3

    # Simulate click, outside data
    event = read_only_mouse_event(
        type='mouse_press',
        position=(-1, -1, -1),
        view_direction=(1, 0, 0),
        dims_displayed=(0, 1, 2),
        dims_point=(0, 0, 0),
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = read_only_mouse_event(
        type='mouse_release',
        position=(-1, -1, -1),
        view_direction=(1, 0, 0),
        dims_displayed=(0, 1, 2),
        dims_point=(0, 0, 0),
    )
    mouse_release_callbacks(layer, event)

    # Undo queue should be empty
    assert len(layer._undo_history) == 0


@pytest.mark.parametrize('mode', ['paint', 'erase', 'fill'])
def test_paint_erase_fill_outside_slice_extent(mode, monkeypatch):
    """Regression test for #9881: interactive painting outside slice extent must not raise IndexError."""
    data = np.zeros((16, 64, 64), dtype=np.uint8)
    layer = Labels(data)
    layer.mode = mode
    layer.selected_label = 2
    # Slice at z=29, outside the layer's 16 slices
    dims = Dims(
        ndim=3,
        ndisplay=2,
        range=((0, 32, 1), (0, 64, 1), (0, 64, 1)),
        point=(29, 0, 0),
    )
    layer._slice_dims(dims)
    assert layer._slice.empty

    notifications = []
    monkeypatch.setattr(
        'napari.layers.labels._labels_mouse_bindings.show_info',
        notifications.append,
    )

    event = read_only_mouse_event(
        type='mouse_press',
        position=(29, 32, 32),
        view_direction=(1, 0, 0),
        dims_displayed=(1, 2),
        dims_point=(29, 32, 32),
    )
    # Must not raise IndexError
    mouse_press_callbacks(layer, event)

    event_rel = read_only_mouse_event(
        type='mouse_release',
        position=(29, 32, 32),
        view_direction=(1, 0, 0),
        dims_displayed=(1, 2),
        dims_point=(29, 32, 32),
    )
    mouse_release_callbacks(layer, event_rel)

    # Notification must have been displayed informing user
    assert len(notifications) == 1
    assert 'has no data on this slice' in notifications[0]
    # Layer data must not be modified
    assert not np.any(layer.data != 0)
    # Undo queue must remain empty
    assert len(layer._undo_history) == 0
