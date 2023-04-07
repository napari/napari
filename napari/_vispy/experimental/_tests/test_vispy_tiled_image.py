import numpy as np
import pytest

from napari._tests.utils import skip_local_popups, skip_on_win_ci
from napari._vispy.experimental.vispy_tiled_image_layer import (
    VispyTiledImageLayer,
)

# Add a loading delay in ms.
SHORT_LOADING_DELAY = 1
LONG_LOADING_DELAY = 250

# Test all dtypes
dtypes = [
    np.dtype(bool),
    np.dtype(np.int8),
    np.dtype(np.uint8),
    np.dtype(np.int16),
    np.dtype(np.uint16),
    np.dtype(np.int32),
    np.dtype(np.uint32),
    np.dtype(np.int64),
    np.dtype(np.uint64),
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.float64),
]


@pytest.mark.async_only
@pytest.mark.skip("NAPARI_OCTREE env var cannot be dynamically set")
@skip_on_win_ci
@skip_local_popups
@pytest.mark.parametrize('dtype', dtypes)
def test_tiled_screenshot(qtbot, monkeypatch, make_napari_viewer, dtype):
    """Test rendering of tiled data with screenshot."""
    # Enable tiled rendering
    monkeypatch.setenv("NAPARI_OCTREE", "1")

    viewer = make_napari_viewer(show=True)
    # Set canvas size to target amount
    viewer.window._qt_viewer.view.canvas.size = (800, 600)

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    data = [100 * np.ones(s, dtype) for s in shapes]
    layer = viewer.add_image(
        data, multiscale=True, contrast_limits=[0, 200], colormap='blue'
    )

    visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]

    # Check visual is a tiled image visual
    assert isinstance(visual, VispyTiledImageLayer)

    # Wait until the chunks have added, ToDo change this to a qtbot.waitSignal
    qtbot.wait(SHORT_LOADING_DELAY)

    # Take the screenshot
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(int)
    target_center = np.array([0, 0, 128, 255], dtype='uint8')
    target_edge = np.array([0, 0, 0, 255], dtype='uint8')
    screen_offset = 3  # Offset is needed as our screenshots have black borders

    np.testing.assert_allclose(screenshot[tuple(center_coord)], target_center)
    np.testing.assert_allclose(
        screenshot[screen_offset, screen_offset], target_edge
    )
    np.testing.assert_allclose(
        screenshot[-screen_offset, -screen_offset], target_edge
    )


@pytest.mark.async_only
@pytest.mark.skip("NAPARI_OCTREE env var cannot be dynamically set")
@skip_on_win_ci
@skip_local_popups
def test_tiled_rgb(qtbot, monkeypatch, make_napari_viewer):
    """Test rgb data works as expected."""
    # Enable tiled rendering
    monkeypatch.setenv("NAPARI_OCTREE", "1")

    viewer = make_napari_viewer(show=True)
    # Set canvas size to target amount
    viewer.window._qt_viewer.view.canvas.size = (800, 600)

    shapes = [(4000, 3000, 3), (2000, 1500, 3), (1000, 750, 3), (500, 375, 3)]
    data = [128 * np.ones(s, np.uint8) for s in shapes]
    layer = viewer.add_image(data, multiscale=True, rgb=True)

    visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]

    # Check visual is a tiled image visual
    assert isinstance(visual, VispyTiledImageLayer)

    # Wait until the chunks have added, ToDo change this to a qtbot.waitSignal
    qtbot.wait(SHORT_LOADING_DELAY)

    # Take the screenshot
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(int)
    target_center = np.array([128, 128, 128, 255], dtype='uint8')
    target_edge = np.array([0, 0, 0, 255], dtype='uint8')
    screen_offset = 3  # Offset is needed as our screenshots have black borders

    # Center pixel should be gray
    np.testing.assert_allclose(screenshot[tuple(center_coord)], target_center)
    np.testing.assert_allclose(
        screenshot[screen_offset, screen_offset], target_edge
    )
    np.testing.assert_allclose(
        screenshot[-screen_offset, -screen_offset], target_edge
    )


@pytest.mark.async_only
@pytest.mark.skip("NAPARI_OCTREE env var cannot be dynamically set")
@skip_on_win_ci
@skip_local_popups
def test_tiled_changing_contrast_limits(
    qtbot, monkeypatch, make_napari_viewer
):
    """Test changing contrast limits of tiled data."""
    # Enable tiled rendering
    monkeypatch.setenv("NAPARI_OCTREE", "1")

    viewer = make_napari_viewer(show=True)
    # Set canvas size to target amount
    viewer.window._qt_viewer.view.canvas.size = (800, 600)

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    data = [np.ones(s, np.uint8) for s in shapes]
    layer = viewer.add_image(
        data, multiscale=True, contrast_limits=[0, 1000], colormap='blue'
    )

    visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]

    # Check visual is a tiled image visual
    assert isinstance(visual, VispyTiledImageLayer)

    # Wait until the chunks have added, ToDo change this to a qtbot.waitSignal
    qtbot.wait(SHORT_LOADING_DELAY)

    # Take the screenshot
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(int)
    target_center = np.array([0, 0, 255, 255], dtype='uint8')
    target_edge = np.array([0, 0, 0, 255], dtype='uint8')
    screen_offset = 3  # Offset is needed as our screenshots have black borders

    # Center pixel should be black as contrast limits are so large
    np.testing.assert_allclose(screenshot[tuple(center_coord)], target_edge)
    np.testing.assert_allclose(
        screenshot[screen_offset, screen_offset], target_edge
    )
    np.testing.assert_allclose(
        screenshot[-screen_offset, -screen_offset], target_edge
    )

    # Make clim data range so center pixel now appears fully saturated
    layer.contrast_limits = [0, 1]

    # Required wait is longer, ToDo change this to a qtbot.waitSignal
    qtbot.wait(LONG_LOADING_DELAY)

    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    np.testing.assert_allclose(screenshot[tuple(center_coord)], target_center)


@pytest.mark.async_only
@pytest.mark.skip("NAPARI_OCTREE env var cannot be dynamically set")
@skip_on_win_ci
@skip_local_popups
def test_tiled_single_scale(qtbot, monkeypatch, make_napari_viewer):
    """Test rgb data works as expected."""
    # Enable tiled rendering
    monkeypatch.setenv("NAPARI_OCTREE", "1")

    viewer = make_napari_viewer(show=True)
    # Set canvas size to target amount
    viewer.window._qt_viewer.view.canvas.size = (800, 600)

    # Add a single scale image.
    layer = viewer.add_image(np.ones((4000, 3000)), contrast_limits=[0, 2])
    # zoom in so as not to load all the data
    viewer.camera.zoom = 0.5

    visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]

    # Check visual is a tiled image visual
    assert isinstance(visual, VispyTiledImageLayer)

    # Wait until the chunks have added, ToDo change this to a qtbot.waitSignal
    # Need an extra long delay here for all tiles to load, including those at
    # edge, as zoomed in.
    qtbot.wait(10 * LONG_LOADING_DELAY)

    # Take the screenshot
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(int)
    target_center = np.array([128, 128, 128, 255], dtype='uint8')
    screen_offset = 3  # Offset is needed as our screenshots have black borders

    # Center pixel should be gray, as should edge as zoomed in
    np.testing.assert_allclose(screenshot[tuple(center_coord)], target_center)
    np.testing.assert_allclose(
        screenshot[screen_offset, screen_offset], target_center
    )
    np.testing.assert_allclose(
        screenshot[-screen_offset, -screen_offset], target_center
    )


@pytest.mark.async_only
@pytest.mark.skip("NAPARI_OCTREE env var cannot be dynamically set")
@skip_on_win_ci
@skip_local_popups
def test_tiled_labels(qtbot, monkeypatch, make_napari_viewer):
    """Test labels data works as expected."""
    # Enable tiled rendering
    monkeypatch.setenv("NAPARI_OCTREE", "1")

    viewer = make_napari_viewer(show=True)
    # Set canvas size to target amount
    viewer.window._qt_viewer.view.canvas.size = (800, 600)

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    data = [np.ones(s, np.uint8) for s in shapes]
    layer = viewer.add_labels(data, multiscale=True, opacity=1)

    visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]

    # Check visual is a tiled image visual
    assert isinstance(visual, VispyTiledImageLayer)

    # Wait until the chunks have added, ToDo change this to a qtbot.waitSignal
    qtbot.wait(SHORT_LOADING_DELAY)

    # Take the screenshot
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(int)
    col = layer.get_color(1)
    target_center = np.array([c * 255 for c in col], dtype='uint8')
    target_edge = np.array([0, 0, 0, 255], dtype='uint8')
    screen_offset = 3  # Offset is needed as our screenshots have black borders

    # Center pixel should be gray
    np.testing.assert_allclose(
        screenshot[tuple(center_coord)], target_center, atol=1
    )
    np.testing.assert_allclose(
        screenshot[screen_offset, screen_offset], target_edge
    )
    np.testing.assert_allclose(
        screenshot[-screen_offset, -screen_offset], target_edge
    )
