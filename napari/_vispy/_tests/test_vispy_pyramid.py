import numpy as np


# def test_corner_value():
#     """Test getting the value of the data at the new position."""
#     shapes = [(40, 20), (20, 10), (10, 5)]
#     np.random.seed(0)
#     data = [np.random.random(s) for s in shapes]
#     layer = Image(data, is_pyramid=True)
#     value = layer.get_value()
#     target_position = (39, 19)
#     target_level = 0
#     layer.data_level = target_level
#     layer._corner_pixels[1] = shapes[target_level] #update requested view
#     layer.refresh()
#
#     # Test position at corner of image
#     layer.position = target_position
#     value = layer.get_value()
#     np.testing.assert_allclose(value, (target_level, data[target_level][target_position]))
#
#     # Test position at outside image
#     layer.position = (40, 20)
#     value = layer.get_value()
#     assert value[1] is None


def test_pyramid(viewer_factory):
    """Test rendering of pyramid data."""
    view, viewer = viewer_factory(show=True)

    shapes = [(4000, 2000), (2000, 1000), (1000, 500), (500, 250)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    _ = viewer.add_image(data, is_pyramid=True)
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
    np.random.seed(0)
    data = [np.ones(s) for s in shapes]
    _ = viewer.add_image(data, is_pyramid=True)

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
    np.random.seed(0)
    data = [np.ones(s) for s in shapes]
    _ = viewer.add_image(data, is_pyramid=True)

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

    np.random.seed(0)
    _ = viewer.add_image(np.ones((4000, 2000)), is_pyramid=True)

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
