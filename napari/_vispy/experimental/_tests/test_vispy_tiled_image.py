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


@pytest.mark.async_only
@skip_on_win_ci
@skip_local_popups
def test_tiled_screenshot(make_napari_viewer):
    """Test rendering of tiled data with screenshot."""
    viewer = make_napari_viewer(show=True)
    # Set canvas size to target amount
    viewer.window.qt_viewer.view.canvas.size = (800, 600)

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    data = [np.ones(s) for s in shapes]
    layer = viewer.add_image(data, multiscale=True, contrast_limits=[0, 1])

    visual = viewer.window.qt_viewer.layer_to_visual[layer]

    # Check visual is a tiled image visual
    assert isinstance(visual, VispyTiledImageLayer)

    # Wait until the some chunks need to be added
    while visual._update_view() == 0:
        return

    # Wait until the no more chunks need to be added
    while visual._update_view() > 0:
        return

    # Take the screenshot
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


@pytest.mark.async_only
@skip_on_win_ci
@skip_local_popups
def test_tiled_changing_contrast_limits(make_napari_viewer):
    """Test changing contrast limits of tiled data."""
    viewer = make_napari_viewer(show=True)
    # Set canvas size to target amount
    viewer.window.qt_viewer.view.canvas.size = (800, 600)

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    data = [np.ones(s) for s in shapes]
    layer = viewer.add_image(data, multiscale=True, contrast_limits=[0, 1])

    visual = viewer.window.qt_viewer.layer_to_visual[layer]

    # Check visual is a tiled image visual
    assert isinstance(visual, VispyTiledImageLayer)

    # Wait until the some chunks need to be added
    while visual._update_view() == 0:
        return

    # Wait until the no more chunks need to be added
    while visual._update_view() > 0:
        return

    # Take the screenshot
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

    # Make clim very large so center pixel now appears black
    layer.contrast_limits = [0, 1000]
    screenshot = viewer.screenshot(canvas_only=True)
    np.testing.assert_allclose(screenshot[tuple(center_coord)], target_edge)
