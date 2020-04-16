import numpy as np


def test_pyramid(viewer_factory):
    """Test rendering of pyramid data."""
    view, viewer = viewer_factory(show=True)

    shapes = [(4000, 2000), (2000, 1000), (1000, 500), (500, 250)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    _ = viewer.add_image(data, is_pyramid=True, contrast_limits=[0, 1])
    layer = viewer.layers[0]

    # Set canvas size to target amount
    view.view.canvas.size = (800, 600)

    # Check that current level is top level of pyramid
    assert layer.data_level == 3
    # Check that full field of view is currently requested
    np.testing.assert_allclose(
        layer.corner_pixels, np.array([[0, 0], np.subtract(shapes[3], 1)])
    )

    # Test value at top left corner of image
    layer.position = (0, 0)
    value = layer.get_value()
    np.testing.assert_allclose(value, (3, data[3][(0, 0)]))

    # Test value at bottom right corner of image
    layer.position = (499, 249)
    value = layer.get_value()
    np.testing.assert_allclose(value, (3, data[3][(499, 249)]))

    # Test value outside image
    layer.position = (500, 250)
    value = layer.get_value()
    assert value[1] is None


def test_pyramid_screenshot(viewer_factory):
    """Test rendering of pyramid data with screenshot."""
    view, viewer = viewer_factory(show=True)

    shapes = [(4000, 2000), (2000, 1000), (1000, 500), (500, 250)]
    data = [np.ones(s) for s in shapes]
    _ = viewer.add_image(data, is_pyramid=True, contrast_limits=[0, 1])

    # Set canvas size to target amount
    view.view.canvas.size = (800, 600)

    screenshot = viewer.screenshot()
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


def test_pyramid_screenshot_zoomed(viewer_factory):
    """Test rendering of pyramid data with screenshot after zoom."""
    view, viewer = viewer_factory(show=True)

    shapes = [(4000, 2000), (2000, 1000), (1000, 500), (500, 250)]
    data = [np.ones(s) for s in shapes]
    _ = viewer.add_image(data, is_pyramid=True, contrast_limits=[0, 1])

    # Set canvas size to target amount
    view.view.canvas.size = (800, 600)

    # Set zoom of camera to show highest resolution tile
    view.view.camera.rect = [1000, 1000, 100, 200]
    view.on_draw(None)

    # Check that current level is bottom level of pyramid
    assert viewer.layers[0].data_level == 0

    screenshot = viewer.screenshot()
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


def test_image_screenshot_zoomed(viewer_factory):
    """Test rendering of image data with screenshot after zoom."""
    view, viewer = viewer_factory(show=True)

    data = np.ones((4000, 2000))
    _ = viewer.add_image(data, is_pyramid=True, contrast_limits=[0, 1])

    # Set canvas size to target amount
    view.view.canvas.size = (800, 600)

    # Set zoom of camera to show highest resolution tile
    view.view.camera.rect = [1000, 1000, 100, 200]
    view.on_draw(None)

    screenshot = viewer.screenshot()
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
