import os
import sys

import numpy as np
import pytest


@pytest.mark.skipif(
    sys.platform.startswith('win') or not os.getenv("CI"),
    reason='Screenshot tests are not supported on napari windows CI.',
)
def test_z_order_adding_removing_images(make_test_viewer):
    """Test z order is correct after adding/ removing images."""
    data = np.ones((10, 10))

    viewer = make_test_viewer(show=True)
    viewer.add_image(data, colormap='red', name='red')
    viewer.add_image(data, colormap='green', name='green')
    viewer.add_image(data, colormap='blue', name='blue')

    # Check that blue is visible
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # Remove and re-add image
    viewer.layers.remove('red')
    viewer.add_image(data, colormap='red', name='red')
    # Check that red is visible
    screenshot = viewer.screenshot(canvas_only=True)
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])

    # Remove two other images
    viewer.layers.remove('green')
    viewer.layers.remove('blue')
    # Check that red is still visible
    screenshot = viewer.screenshot(canvas_only=True)
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])

    # Add two other layers back
    viewer.add_image(data, colormap='green', name='green')
    viewer.add_image(data, colormap='blue', name='blue')
    # Check that blue is visible
    screenshot = viewer.screenshot(canvas_only=True)
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # Hide blue
    viewer.layers['blue'].visible = False
    # Check that green is visible. Note this assert was failing before #1463
    screenshot = viewer.screenshot(canvas_only=True)
    np.testing.assert_almost_equal(screenshot[center], [0, 255, 0, 255])


@pytest.mark.skipif(
    sys.platform.startswith('win') or not os.getenv("CI"),
    reason='Screenshot tests are not supported on napari windows CI.',
)
def test_z_order_images(make_test_viewer):
    """Test changing order of images changes z order in display."""
    data = np.ones((10, 10))

    viewer = make_test_viewer(show=True)
    viewer.add_image(data, colormap='red')
    viewer.add_image(data, colormap='blue')
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    viewer.layers[0, 1] = viewer.layers[1, 0]
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that red is now visible
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])


@pytest.mark.skipif(
    sys.platform.startswith('win') or not os.getenv("CI"),
    reason='Screenshot tests are not supported on napari windows CI.',
)
def test_z_order_image_points(make_test_viewer):
    """Test changing order of image and points changes z order in display."""
    data = np.ones((10, 10))

    viewer = make_test_viewer(show=True)
    viewer.add_image(data, colormap='red')
    viewer.add_points([5, 5], face_color='blue', size=10)
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    viewer.layers[0, 1] = viewer.layers[1, 0]
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that red is now visible
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])


@pytest.mark.skipif(
    sys.platform.startswith('win') or not os.getenv("CI"),
    reason='Screenshot tests are not supported on napari windows CI.',
)
def test_z_order_images_after_ndisplay(make_test_viewer):
    """Test z order of images remanins constant after chaning ndisplay."""
    data = np.ones((10, 10))

    viewer = make_test_viewer(show=True)
    viewer.add_image(data, colormap='red')
    viewer.add_image(data, colormap='blue')
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # Switch to 3D rendering
    viewer.dims.ndisplay = 3
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is still visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # Switch back to 2D rendering
    viewer.dims.ndisplay = 2
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is still visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])


@pytest.mark.skipif(
    sys.platform.startswith('win') or not os.getenv("CI"),
    reason='Screenshot tests are not supported on napari windows CI.',
)
def test_z_order_image_points_after_ndisplay(make_test_viewer):
    """Test z order of image and points remanins constant after chaning ndisplay."""
    data = np.ones((10, 10))

    viewer = make_test_viewer(show=True)
    viewer.add_image(data, colormap='red')
    viewer.add_points([5, 5], face_color='blue', size=10)
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # Switch to 3D rendering
    viewer.dims.ndisplay = 3
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is still visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # Switch back to 2D rendering
    viewer.dims.ndisplay = 2
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is still visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])


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
