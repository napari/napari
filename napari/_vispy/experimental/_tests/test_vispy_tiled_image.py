import os
import sys

import numpy as np
import pytest

from napari._vispy.experimental.vispy_tiled_image_layer import (
    VispyTiledImageLayer,
)

skip_on_win_ci = pytest.mark.skipif(
    sys.platform.startswith('win') and os.getenv('CI', '0') != '0',
    reason='Screenshot tests are not supported on windows CI.',
)
skip_local_popups = pytest.mark.skipif(
    not os.getenv('CI') and os.getenv('NAPARI_POPUP_TESTS', '0') == '0',
    reason='Tests requiring GUI windows are skipped locally by default.',
)

# Add a loading delay in ms, any number > 0 seems to work.
LOADING_DELAY = 1

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
@skip_on_win_ci
@skip_local_popups
@pytest.mark.parametrize('dtype', dtypes)
def test_tiled_screenshot(qtbot, make_napari_viewer, dtype):
    """Test rendering of tiled data with screenshot."""
    viewer = make_napari_viewer(show=True)
    # Set canvas size to target amount
    viewer.window.qt_viewer.view.canvas.size = (800, 600)

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    data = [np.ones(s, dtype) for s in shapes]
    layer = viewer.add_image(
        data, multiscale=True, contrast_limits=[0, 1], colormap='blue'
    )

    visual = viewer.window.qt_viewer.layer_to_visual[layer]

    # Check visual is a tiled image visual
    assert isinstance(visual, VispyTiledImageLayer)

    # Wait until the chunks have added
    qtbot.wait(LOADING_DELAY)

    # Take the screenshot
    screenshot = viewer.screenshot(canvas_only=True)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(np.int)
    target_center = np.array([0, 0, 255, 255], dtype='uint8')
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
@skip_on_win_ci
@skip_local_popups
def test_tiled_changing_contrast_limits(qtbot, make_napari_viewer):
    """Test changing contrast limits of tiled data."""
    viewer = make_napari_viewer(show=True)
    # Set canvas size to target amount
    viewer.window.qt_viewer.view.canvas.size = (800, 600)

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    data = [np.ones(s, np.uint8) for s in shapes]
    layer = viewer.add_image(
        data, multiscale=True, contrast_limits=[0, 1000], colormap='blue'
    )

    visual = viewer.window.qt_viewer.layer_to_visual[layer]

    # Check visual is a tiled image visual
    assert isinstance(visual, VispyTiledImageLayer)

    # Wait until the chunks have added
    qtbot.wait(LOADING_DELAY)

    # Take the screenshot
    screenshot = viewer.screenshot(canvas_only=True)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(np.int)
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

    screenshot = viewer.screenshot(canvas_only=True)
    np.testing.assert_allclose(screenshot[tuple(center_coord)], target_center)
