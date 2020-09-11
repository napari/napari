import os
import sys

import numpy as np
import pytest


def test_multiscale(make_test_viewer):
    """Test rendering of multiscale data."""
    viewer = make_test_viewer()

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    _ = viewer.add_image(data, multiscale=True, contrast_limits=[0, 1])
    layer = viewer.layers[0]

    # Set canvas size to target amount
    viewer.window.qt_viewer.view.canvas.size = (800, 600)
    list(viewer.window.qt_viewer.layer_to_visual.values())[0].on_draw(None)

    # Check that current level is first large enough to fill the canvas with
    # a greater than one pixel depth
    assert layer.data_level == 2

    # Check that full field of view is currently requested
    assert np.all(layer.corner_pixels[0] <= [0, 0])
    assert np.all(layer.corner_pixels[1] >= np.subtract(shapes[2], 1))

    # Test value at top left corner of image
    layer.position = (0, 0)
    value = layer.get_value()
    np.testing.assert_allclose(value, (2, data[2][(0, 0)]))

    # Test value at bottom right corner of image
    layer.position = (999, 749)
    value = layer.get_value()
    np.testing.assert_allclose(value, (2, data[2][(999, 749)]))

    # Test value outside image
    layer.position = (1000, 750)
    value = layer.get_value()
    assert value[1] is None


def test_3D_multiscale_image(make_test_viewer):
    """Test rendering of 3D multiscale image uses lowest resolution."""
    viewer = make_test_viewer()

    data = [np.random.random((128,) * 3), np.random.random((64,) * 3)]
    viewer.add_image(data)

    # Check that this doesn't crash.
    viewer.dims.ndisplay = 3

    # Check lowest resolution is used
    assert viewer.layers[0].data_level == 1

    # Note that draw command must be explicitly triggered in our tests
    list(viewer.window.qt_viewer.layer_to_visual.values())[0].on_draw(None)


@pytest.mark.skipif(
    sys.platform.startswith('win') or not os.getenv("CI"),
    reason='Screenshot tests are not supported on napari windows CI.',
)
def test_multiscale_screenshot(make_test_viewer):
    """Test rendering of multiscale data with screenshot."""
    viewer = make_test_viewer(show=True)

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    data = [np.ones(s) for s in shapes]
    _ = viewer.add_image(data, multiscale=True, contrast_limits=[0, 1])

    # Set canvas size to target amount
    viewer.window.qt_viewer.view.canvas.size = (800, 600)

    screenshot = viewer.screenshot(canvas_only=True)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(np.int)
    target_center = np.array([255, 255, 255, 255], dtype='uint8')
    target_edge = np.array([0, 0, 0, 255], dtype='uint8')
    screen_offset = 3  # Offset is needed as our screenshots have black borders

    np.testing.assert_allclose(screenshot[tuple(center_coord)], target_center)
    np.testing.assert_allclose(
        screenshot[screen_offset, screen_offset], target_edge
    )
    np.testing.assert_allclose(
        screenshot[-screen_offset, -screen_offset], target_edge
    )


@pytest.mark.skipif(
    sys.platform.startswith('win') or not os.getenv("CI"),
    reason='Screenshot tests are not supported on napari windows CI.',
)
def test_multiscale_screenshot_zoomed(make_test_viewer):
    """Test rendering of multiscale data with screenshot after zoom."""
    viewer = make_test_viewer(show=True)
    view = viewer.window.qt_viewer

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    data = [np.ones(s) for s in shapes]
    _ = viewer.add_image(data, multiscale=True, contrast_limits=[0, 1])

    # Set canvas size to target amount
    view.view.canvas.size = (800, 600)

    # Set zoom of camera to show highest resolution tile
    view.view.camera.rect = [1000, 1000, 200, 150]
    list(view.layer_to_visual.values())[0].on_draw(None)

    # Check that current level is bottom level of multiscale
    assert viewer.layers[0].data_level == 0

    screenshot = viewer.screenshot(canvas_only=True)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(np.int)
    target_center = np.array([255, 255, 255, 255], dtype='uint8')
    screen_offset = 3  # Offset is needed as our screenshots have black borders

    np.testing.assert_allclose(screenshot[tuple(center_coord)], target_center)
    np.testing.assert_allclose(
        screenshot[screen_offset, screen_offset], target_center
    )
    np.testing.assert_allclose(
        screenshot[-screen_offset, -screen_offset], target_center
    )


@pytest.mark.skipif(
    sys.platform.startswith('win') or not os.getenv("CI"),
    reason='Screenshot tests are not supported on napari windows CI.',
)
def test_image_screenshot_zoomed(make_test_viewer):
    """Test rendering of image data with screenshot after zoom."""
    viewer = make_test_viewer(show=True)
    view = viewer.window.qt_viewer

    data = np.ones((4000, 3000))
    _ = viewer.add_image(data, multiscale=False, contrast_limits=[0, 1])

    # Set canvas size to target amount
    view.view.canvas.size = (800, 600)

    # Set zoom of camera to show highest resolution tile
    view.view.camera.rect = [1000, 1000, 200, 150]
    list(view.layer_to_visual.values())[0].on_draw(None)

    screenshot = viewer.screenshot(canvas_only=True)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(np.int)
    target_center = np.array([255, 255, 255, 255], dtype='uint8')
    screen_offset = 3  # Offset is needed as our screenshots have black borders

    np.testing.assert_allclose(screenshot[tuple(center_coord)], target_center)
    np.testing.assert_allclose(
        screenshot[screen_offset, screen_offset], target_center
    )
    np.testing.assert_allclose(
        screenshot[-screen_offset, -screen_offset], target_center
    )
