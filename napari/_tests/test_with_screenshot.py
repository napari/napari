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


@pytest.mark.skipif(
    sys.platform.startswith('win') or not os.getenv("CI"),
    reason='Screenshot tests are not supported on napari windows CI.',
)
def test_grid_mode(make_test_viewer):
    """Test changing gamma changes rendering."""
    viewer = make_test_viewer(show=True)

    # Add images
    data = np.ones((6, 15, 15))
    viewer.add_image(data, channel_axis=0, blending='translucent')

    assert np.all(viewer.grid_size == (1, 1))
    assert viewer.grid_stride == 1
    translations = [layer.translate_grid for layer in viewer.layers]
    expected_translations = np.zeros((6, 2))
    np.testing.assert_allclose(translations, expected_translations)

    # check screenshot
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # enter grid view
    viewer.grid_view()
    assert np.all(viewer.grid_size == (3, 3))
    assert viewer.grid_stride == 1
    translations = [layer.translate_grid for layer in viewer.layers]
    expected_translations = [
        [0, 0],
        [0, 15],
        [0, 30],
        [15, 0],
        [15, 15],
        [15, 30],
    ]
    np.testing.assert_allclose(translations, expected_translations[::-1])

    # check screenshot
    screenshot = viewer.screenshot(canvas_only=True)
    # sample 6 squares of the grid and check they have right colors
    pos = [
        (1 / 3, 1 / 3),
        (1 / 3, 1 / 2),
        (1 / 3, 2 / 3),
        (1 / 2, 1 / 3),
        (1 / 2, 1 / 2),
        (1 / 2, 2 / 3),
    ]
    # BGRMYC color order
    color = [
        [0, 0, 255, 255],
        [0, 255, 0, 255],
        [255, 0, 0, 255],
        [255, 0, 255, 255],
        [255, 255, 0, 255],
        [0, 255, 255, 255],
    ]
    for c, p in zip(color, pos):
        coord = tuple(
            np.round(np.multiply(screenshot.shape[:2], p)).astype(int)
        )
        np.testing.assert_almost_equal(screenshot[coord], c)

    # reorder layers
    viewer.layers[0, 5] = viewer.layers[5, 0]

    # check screenshot
    screenshot = viewer.screenshot(canvas_only=True)
    # sample 6 squares of the grid and check they have right colors
    pos = [
        (1 / 3, 1 / 3),
        (1 / 3, 1 / 2),
        (1 / 3, 2 / 3),
        (1 / 2, 1 / 3),
        (1 / 2, 1 / 2),
        (1 / 2, 2 / 3),
    ]
    # CGRMYB color order
    color = [
        [0, 255, 255, 255],
        [0, 255, 0, 255],
        [255, 0, 0, 255],
        [255, 0, 255, 255],
        [255, 255, 0, 255],
        [0, 0, 255, 255],
    ]
    for c, p in zip(color, pos):
        coord = tuple(
            np.round(np.multiply(screenshot.shape[:2], p)).astype(int)
        )
        np.testing.assert_almost_equal(screenshot[coord], c)

    # retun to stack view
    viewer.stack_view()
    assert np.all(viewer.grid_size == (1, 1))
    assert viewer.grid_stride == 1
    translations = [layer.translate_grid for layer in viewer.layers]
    expected_translations = np.zeros((6, 2))
    np.testing.assert_allclose(translations, expected_translations)

    # check screenshot
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    np.testing.assert_almost_equal(screenshot[center], [0, 255, 255, 255])


@pytest.mark.skipif(
    sys.platform.startswith('win') or not os.getenv("CI"),
    reason='Screenshot tests are not supported on napari windows CI.',
)
def test_changing_image_attenuation(make_test_viewer):
    """Test changing attenuation value changes rendering."""
    data = np.zeros((100, 10, 10))
    data[-1] = 1

    viewer = make_test_viewer(show=True)
    viewer.dims.ndisplay = 3
    viewer.add_image(data, contrast_limits=[0, 1])
    viewer.layers[0].rendering = 'attenuated_mip'

    viewer.layers[0].attenuation = 0.5
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that rendering has not been attenuated
    assert screenshot[center + (0,)] > 80

    viewer.layers[0].attenuation = 0.02
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that rendering has been attenuated
    assert screenshot[center + (0,)] < 60
