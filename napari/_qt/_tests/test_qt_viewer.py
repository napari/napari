import os
from dataclasses import dataclass
from typing import List
from unittest import mock

import numpy as np
import pytest
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QMessageBox

from napari._tests.utils import (
    add_layer_by_type,
    check_viewer_functioning,
    layer_test_data,
)
from napari.utils.interactions import mouse_press_callbacks
from napari.utils.io import imread


def test_qt_viewer(make_napari_viewer):
    """Test instantiating viewer."""
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer

    assert viewer.title == 'napari'
    assert view.viewer == viewer
    # Check no console is present before it is requested
    assert view._console is None

    assert len(viewer.layers) == 0
    assert view.layers.model().rowCount() == 0

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_qt_viewer_with_console(make_napari_viewer):
    """Test instantiating console from viewer."""
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer
    # Check no console is present before it is requested
    assert view._console is None
    # Check console is created when requested
    assert view.console is not None
    assert view.dockConsole.widget() is view.console


def test_qt_viewer_toggle_console(make_napari_viewer):
    """Test instantiating console from viewer."""
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer
    # Check no console is present before it is requested
    assert view._console is None
    # Check console has been created when it is supposed to be shown
    view.toggle_console_visibility(None)
    assert view._console is not None
    assert view.dockConsole.widget() is view.console


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_add_layer(make_napari_viewer, layer_class, data, ndim):
    viewer = make_napari_viewer(ndisplay=int(np.clip(ndim, 2, 3)))
    view = viewer.window.qt_viewer

    add_layer_by_type(viewer, layer_class, data)
    check_viewer_functioning(viewer, view, data, ndim)


def test_new_labels(make_napari_viewer):
    """Test adding new labels layer."""
    # Add labels to empty viewer
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer

    viewer._new_labels()
    assert np.max(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.model().rowCount() == len(viewer.layers)

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Add labels with image already present
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer._new_labels()
    assert np.max(viewer.layers[1].data) == 0
    assert len(viewer.layers) == 2
    assert view.layers.model().rowCount() == len(viewer.layers)

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_new_points(make_napari_viewer):
    """Test adding new points layer."""
    # Add labels to empty viewer
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer

    viewer.add_points()
    assert len(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.model().rowCount() == len(viewer.layers)

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Add points with image already present
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer.add_points()
    assert len(viewer.layers[1].data) == 0
    assert len(viewer.layers) == 2
    assert view.layers.model().rowCount() == len(viewer.layers)

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_new_shapes_empty_viewer(make_napari_viewer):
    """Test adding new shapes layer."""
    # Add labels to empty viewer
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer

    viewer.add_shapes()
    assert len(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.model().rowCount() == len(viewer.layers)

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Add points with image already present
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer.add_shapes()
    assert len(viewer.layers[1].data) == 0
    assert len(viewer.layers) == 2
    assert view.layers.model().rowCount() == len(viewer.layers)

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_z_order_adding_removing_images(make_napari_viewer):
    """Test z order is correct after adding/ removing images."""
    data = np.ones((10, 10))

    viewer = make_napari_viewer()
    vis = viewer.window.qt_viewer.layer_to_visual
    viewer.add_image(data, colormap='red', name='red')
    viewer.add_image(data, colormap='green', name='green')
    viewer.add_image(data, colormap='blue', name='blue')
    order = [vis[x].order for x in viewer.layers]
    np.testing.assert_almost_equal(order, list(range(len(viewer.layers))))

    # Remove and re-add image
    viewer.layers.remove('red')
    order = [vis[x].order for x in viewer.layers]
    np.testing.assert_almost_equal(order, list(range(len(viewer.layers))))
    viewer.add_image(data, colormap='red', name='red')
    order = [vis[x].order for x in viewer.layers]
    np.testing.assert_almost_equal(order, list(range(len(viewer.layers))))

    # Remove two other images
    viewer.layers.remove('green')
    viewer.layers.remove('blue')
    order = [vis[x].order for x in viewer.layers]
    np.testing.assert_almost_equal(order, list(range(len(viewer.layers))))

    # Add two other layers back
    viewer.add_image(data, colormap='green', name='green')
    viewer.add_image(data, colormap='blue', name='blue')
    order = [vis[x].order for x in viewer.layers]
    np.testing.assert_almost_equal(order, list(range(len(viewer.layers))))


def test_screenshot(make_napari_viewer):
    "Test taking a screenshot"
    viewer = make_napari_viewer()

    np.random.seed(0)
    # Add image
    data = np.random.random((10, 15))
    viewer.add_image(data)

    # Add labels
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)

    # Add points
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)

    # Add vectors
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)

    # Add shapes
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)

    # Take screenshot
    screenshot = viewer.window.qt_viewer.screenshot(flash=False)
    assert screenshot.ndim == 3


@pytest.mark.skip("new approach")
def test_screenshot_dialog(make_napari_viewer, tmpdir):
    """Test save screenshot functionality."""
    viewer = make_napari_viewer()

    np.random.seed(0)
    # Add image
    data = np.random.random((10, 15))
    viewer.add_image(data)

    # Add labels
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)

    # Add points
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)

    # Add vectors
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)

    # Add shapes
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)

    # Save screenshot
    input_filepath = os.path.join(tmpdir, 'test-save-screenshot')
    mock_return = (input_filepath, '')
    with mock.patch('napari._qt.qt_viewer.QFileDialog') as mocker, mock.patch(
        'napari._qt.qt_viewer.QMessageBox'
    ) as mocker2:
        mocker.getSaveFileName.return_value = mock_return
        mocker2.warning.return_value = QMessageBox.Yes
        viewer.window.qt_viewer._screenshot_dialog()
    # Assert behaviour is correct
    expected_filepath = input_filepath + '.png'  # add default file extension
    assert os.path.exists(expected_filepath)
    output_data = imread(expected_filepath)
    expected_data = viewer.window.qt_viewer.screenshot(flash=False)
    assert np.allclose(output_data, expected_data)


@pytest.mark.parametrize(
    "dtype", ['int8', 'uint8', 'int16', 'uint16', 'float32']
)
def test_qt_viewer_data_integrity(make_napari_viewer, dtype):
    """Test that the viewer doesn't change the underlying array."""

    image = np.random.rand(10, 32, 32)
    image *= 200 if dtype.endswith('8') else 2 ** 14
    image = image.astype(dtype)
    imean = image.mean()

    viewer = make_napari_viewer()

    viewer.add_image(image.copy())
    datamean = viewer.layers[0].data.mean()
    assert datamean == imean
    # toggle dimensions
    viewer.dims.ndisplay = 3
    datamean = viewer.layers[0].data.mean()
    assert datamean == imean
    # back to 2D
    viewer.dims.ndisplay = 2
    datamean = viewer.layers[0].data.mean()
    assert datamean == imean


def test_points_layer_display_correct_slice_on_scale(make_napari_viewer):
    viewer = make_napari_viewer()
    data = np.zeros((60, 60, 60))
    viewer.add_image(data, scale=[0.29, 0.26, 0.26])
    pts = viewer.add_points(name='test', size=1, ndim=3)
    pts.add((8.7, 0, 0))
    viewer.dims.set_point(0, 30 * 0.29)  # middle plane
    layer = viewer.layers[1]
    indices, scale = layer._slice_data(layer._slice_indices)
    np.testing.assert_equal(indices, [0])


def test_qt_viewer_clipboard_with_flash(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    # make sure clipboard is empty
    QGuiApplication.clipboard().clear()
    clipboard_image = QGuiApplication.clipboard().image()
    assert clipboard_image.isNull()

    # capture screenshot
    viewer.window.qt_viewer.clipboard(flash=True)
    clipboard_image = QGuiApplication.clipboard().image()
    assert not clipboard_image.isNull()

    # ensure the flash effect is applied
    assert viewer.window.qt_viewer._canvas_overlay.graphicsEffect() is not None
    assert hasattr(viewer.window.qt_viewer._canvas_overlay, "_flash_animation")
    qtbot.wait(500)  # wait for the animation to finish
    assert viewer.window.qt_viewer._canvas_overlay.graphicsEffect() is None
    assert not hasattr(
        viewer.window.qt_viewer._canvas_overlay, "_flash_animation"
    )

    # clear clipboard and grab image from application view
    QGuiApplication.clipboard().clear()
    clipboard_image = QGuiApplication.clipboard().image()
    assert clipboard_image.isNull()

    # capture screenshot of the entire window
    viewer.window.clipboard(flash=True)
    clipboard_image = QGuiApplication.clipboard().image()
    assert not clipboard_image.isNull()

    # ensure the flash effect is applied
    assert viewer.window._qt_window.graphicsEffect() is not None
    assert hasattr(viewer.window._qt_window, "_flash_animation")
    qtbot.wait(500)  # wait for the animation to finish
    assert viewer.window._qt_window.graphicsEffect() is None
    assert not hasattr(viewer.window._qt_window, "_flash_animation")


def test_qt_viewer_clipboard_without_flash(make_napari_viewer):
    viewer = make_napari_viewer()
    # make sure clipboard is empty
    QGuiApplication.clipboard().clear()
    clipboard_image = QGuiApplication.clipboard().image()
    assert clipboard_image.isNull()

    # capture screenshot
    viewer.window.qt_viewer.clipboard(flash=False)
    clipboard_image = QGuiApplication.clipboard().image()
    assert not clipboard_image.isNull()

    # ensure the flash effect is not applied
    assert viewer.window.qt_viewer._canvas_overlay.graphicsEffect() is None
    assert not hasattr(
        viewer.window.qt_viewer._canvas_overlay, "_flash_animation"
    )

    # clear clipboard and grab image from application view
    QGuiApplication.clipboard().clear()
    clipboard_image = QGuiApplication.clipboard().image()
    assert clipboard_image.isNull()

    # capture screenshot of the entire window
    viewer.window.clipboard(flash=False)
    clipboard_image = QGuiApplication.clipboard().image()
    assert not clipboard_image.isNull()

    # ensure the flash effect is not applied
    assert viewer.window._qt_window.graphicsEffect() is None
    assert not hasattr(viewer.window._qt_window, "_flash_animation")


def test_active_keybindings(make_napari_viewer):
    """Test instantiating viewer."""
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer

    # Check only keybinding is Viewer
    assert len(view._key_map_handler.keymap_providers) == 1
    assert view._key_map_handler.keymap_providers[0] == viewer

    # Add a layer and check it is keybindings are active
    data = np.random.random((10, 15))
    layer_image = viewer.add_image(data)
    assert viewer.layers.selection.active == layer_image
    assert len(view._key_map_handler.keymap_providers) == 2
    assert view._key_map_handler.keymap_providers[0] == layer_image

    # Add a layer and check it is keybindings become active
    layer_image_2 = viewer.add_image(data)
    assert viewer.layers.selection.active == layer_image_2
    assert len(view._key_map_handler.keymap_providers) == 2
    assert view._key_map_handler.keymap_providers[0] == layer_image_2

    # Change active layer and check it is keybindings become active
    viewer.layers.selection.active = layer_image
    assert viewer.layers.selection.active == layer_image
    assert len(view._key_map_handler.keymap_providers) == 2
    assert view._key_map_handler.keymap_providers[0] == layer_image


@dataclass
class MouseEvent:
    # mock mouse event class
    pos: List[int]


def test_process_mouse_event(make_napari_viewer):
    """Test that the correct properties are added to the
    MouseEvent by _process_mouse_events.
    """
    # make a mock mouse event
    new_pos = [25, 25]
    mouse_event = MouseEvent(
        pos=new_pos,
    )
    data = np.zeros((5, 20, 20, 20), dtype=int)
    data[1, 0:10, 0:10, 0:10] = 1

    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer
    labels = viewer.add_labels(data, scale=(1, 2, 1, 1), translate=(5, 5, 5))

    @labels.mouse_drag_callbacks.append
    def on_click(layer, event):
        np.testing.assert_almost_equal(event.view_direction, [0, 1, 0, 0])
        np.testing.assert_array_equal(event.dims_displayed, [1, 2, 3])
        assert event.dims_point[0] == 0

        expected_position = view._map_canvas2world(new_pos)
        np.testing.assert_almost_equal(expected_position, list(event.position))

    viewer.dims.ndisplay = 3
    view._process_mouse_event(mouse_press_callbacks, mouse_event)
