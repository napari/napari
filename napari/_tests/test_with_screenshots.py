import numpy as np
import os
import sys
import pytest


@pytest.mark.skipif(
    sys.platform.startswith('win') or not os.getenv("CI"),
    reason='Screenshot tests are not supported on napari windows CI.',
)
def test_changing_image_colormap(make_test_viewer):
    """Test changing colormap changes rendering."""
    viewer = make_test_viewer(show=True)

    data = np.ones((20, 20, 20))
    layer = viewer.add_image(data, contrast_limits=[0, 1])

    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    np.testing.assert_almost_equal(screenshot[center], [255, 255, 255, 255])

    layer.colormap = 'red'
    screenshot = viewer.screenshot(canvas_only=True)
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])

    viewer.dims.ndisplay = 3
    screenshot = viewer.screenshot(canvas_only=True)
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])

    layer.colormap = 'blue'
    screenshot = viewer.screenshot(canvas_only=True)
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    viewer.dims.ndisplay = 2
    screenshot = viewer.screenshot(canvas_only=True)
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])


@pytest.mark.skipif(
    sys.platform.startswith('win') or not os.getenv("CI"),
    reason='Screenshot tests are not supported on napari windows CI.',
)
def test_changing_image_gamma(make_test_viewer):
    """Test changing gamma changes rendering."""
    viewer = make_test_viewer(show=True)

    data = np.ones((20, 20, 20))
    layer = viewer.add_image(data, contrast_limits=[0, 2])

    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    assert screenshot[center + (0,)] == 128

    layer.gamma = 0.1
    screenshot = viewer.screenshot(canvas_only=True)
    assert screenshot[center + (0,)] > 230

    viewer.dims.ndisplay = 3
    screenshot = viewer.screenshot(canvas_only=True)
    assert screenshot[center + (0,)] > 230

    layer.gamma = 1.9
    screenshot = viewer.screenshot(canvas_only=True)
    assert screenshot[center + (0,)] < 80

    viewer.dims.ndisplay = 2
    screenshot = viewer.screenshot(canvas_only=True)
    assert screenshot[center + (0,)] < 80
