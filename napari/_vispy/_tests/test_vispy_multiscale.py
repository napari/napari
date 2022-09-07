import numpy as np

from napari._tests.utils import skip_local_popups, skip_on_win_ci


def test_multiscale(make_napari_viewer):
    """Test rendering of multiscale data."""
    viewer = make_napari_viewer()

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    _ = viewer.add_image(data, multiscale=True, contrast_limits=[0, 1])
    layer = viewer.layers[0]

    # Set canvas size to target amount
    viewer.window._qt_viewer.view.canvas.size = (800, 600)
    viewer.window._qt_viewer.on_draw(None)

    # Check that current level is first large enough to fill the canvas with
    # a greater than one pixel depth
    assert layer.data_level == 2

    # Check that full field of view is currently requested
    assert np.all(layer.corner_pixels[0] <= [0, 0])
    assert np.all(layer.corner_pixels[1] >= np.subtract(shapes[2], 1))

    # Test value at top left corner of image
    viewer.cursor.position = (0, 0)
    value = layer.get_value(viewer.cursor.position, world=True)
    np.testing.assert_allclose(value, (2, data[2][(0, 0)]))

    # Test value at bottom right corner of image
    viewer.cursor.position = (3995, 2995)
    value = layer.get_value(viewer.cursor.position, world=True)
    np.testing.assert_allclose(value, (2, data[2][(999, 749)]))

    # Test value outside image
    viewer.cursor.position = (4000, 3000)
    value = layer.get_value(viewer.cursor.position, world=True)
    assert value[1] is None


def test_3D_multiscale_image(make_napari_viewer):
    """Test rendering of 3D multiscale image uses lowest resolution."""
    viewer = make_napari_viewer()

    data = [np.random.random((128,) * 3), np.random.random((64,) * 3)]
    viewer.add_image(data)

    # Check that this doesn't crash.
    viewer.dims.ndisplay = 3

    # Check lowest resolution is used
    assert viewer.layers[0].data_level == 1

    # Note that draw command must be explicitly triggered in our tests
    viewer.window._qt_viewer.on_draw(None)


@skip_on_win_ci
@skip_local_popups
def test_multiscale_screenshot(make_napari_viewer):
    """Test rendering of multiscale data with screenshot."""
    viewer = make_napari_viewer(show=True)

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    data = [np.ones(s) for s in shapes]
    _ = viewer.add_image(data, multiscale=True, contrast_limits=[0, 1])

    # Set canvas size to target amount
    viewer.window._qt_viewer.view.canvas.size = (800, 600)

    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(int)
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


@skip_on_win_ci
@skip_local_popups
def test_multiscale_screenshot_zoomed(make_napari_viewer):
    """Test rendering of multiscale data with screenshot after zoom."""
    viewer = make_napari_viewer(show=True)
    view = viewer.window._qt_viewer

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    data = [np.ones(s) for s in shapes]
    _ = viewer.add_image(data, multiscale=True, contrast_limits=[0, 1])

    # Set canvas size to target amount
    view.view.canvas.size = (800, 600)

    # Set zoom of camera to show highest resolution tile
    view.view.camera.rect = [1000, 1000, 200, 150]
    viewer.window._qt_viewer.on_draw(None)

    # Check that current level is bottom level of multiscale
    assert viewer.layers[0].data_level == 0

    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(int)
    target_center = np.array([255, 255, 255, 255], dtype='uint8')
    screen_offset = 3  # Offset is needed as our screenshots have black borders

    # for whatever reason this is the only test where the border is 6px on hi DPI.
    # if the 6 by 6 corner is all black assume we have a 6px border.
    if not np.allclose(screenshot[:6, :6], np.array([0, 0, 0, 255])):
        screen_offset = 6  # Hi DPI
    np.testing.assert_allclose(screenshot[tuple(center_coord)], target_center)
    np.testing.assert_allclose(
        screenshot[screen_offset, screen_offset], target_center
    )
    np.testing.assert_allclose(
        screenshot[-screen_offset, -screen_offset], target_center
    )


@skip_on_win_ci
@skip_local_popups
def test_image_screenshot_zoomed(make_napari_viewer):
    """Test rendering of image data with screenshot after zoom."""
    viewer = make_napari_viewer(show=True)
    view = viewer.window._qt_viewer

    data = np.ones((4000, 3000))
    _ = viewer.add_image(data, multiscale=False, contrast_limits=[0, 1])

    # Set canvas size to target amount
    view.view.canvas.size = (800, 600)

    # Set zoom of camera to show highest resolution tile
    view.view.camera.rect = [1000, 1000, 200, 150]
    viewer.window._qt_viewer.on_draw(None)

    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(int)
    target_center = np.array([255, 255, 255, 255], dtype='uint8')
    screen_offset = 3  # Offset is needed as our screenshots have black borders

    np.testing.assert_allclose(screenshot[tuple(center_coord)], target_center)
    np.testing.assert_allclose(
        screenshot[screen_offset, screen_offset], target_center
    )
    np.testing.assert_allclose(
        screenshot[-screen_offset, -screen_offset], target_center
    )


@skip_on_win_ci
@skip_local_popups
def test_multiscale_zoomed_out(make_napari_viewer):
    """See https://github.com/napari/napari/issues/4781"""
    # Need to show viewer to ensure that pixel_scale and physical_size
    # get set appropriately.
    viewer = make_napari_viewer(show=True)
    shapes = [(3200, 3200), (1600, 1600), (800, 800)]
    data = [np.empty(s) for s in shapes]
    layer = viewer.add_image(data, multiscale=True)
    qt_viewer = viewer.window._qt_viewer
    # Canvas size is in screen pixels.
    qt_viewer.canvas.size = (1600, 1600)
    # The camera rect is (left, top, width, height) in scene coordinates.
    # In this case scene coordinates are the same as data/world coordinates
    # the layer is 2D and data-to-world is identity.
    # We pick a camera rect size that is much bigger than the data extent
    # to simulate being zoomed out in the viewer.
    camera_rect_size = 34000
    camera_rect_center = 1599.5
    camera_rect_start = camera_rect_center - (camera_rect_size / 2)
    qt_viewer.view.camera.rect = (
        camera_rect_start,
        camera_rect_start,
        camera_rect_size,
        camera_rect_size,
    )

    qt_viewer.on_draw(None)

    assert layer.data_level == 2


@skip_on_win_ci
@skip_local_popups
def test_5D_multiscale(make_napari_viewer):
    """Test 5D multiscale data."""
    # Show must be true to trigger multiscale draw and corner estimation
    viewer = make_napari_viewer(show=True)
    view = viewer.window._qt_viewer
    view.set_welcome_visible(False)
    shapes = [(1, 2, 5, 20, 20), (1, 2, 5, 10, 10), (1, 2, 5, 5, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    layer = viewer.add_image(data, multiscale=True)
    assert layer.data == data
    assert layer.multiscale is True
    assert layer.ndim == len(shapes[0])


@skip_on_win_ci
@skip_local_popups
def test_multiscale_flipped_axes(make_napari_viewer):
    """Check rendering of multiscale images with negative scale values.

    See https://github.com/napari/napari/issues/3057
    """
    viewer = make_napari_viewer(show=True)

    shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
    data = [np.ones(s) for s in shapes]
    # this used to crash, see issue #3057
    _ = viewer.add_image(
        data, multiscale=True, contrast_limits=[0, 1], scale=(-1, 1)
    )


@skip_on_win_ci
@skip_local_popups
def test_multiscale_rotated_image(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    sizes = [4000 // i for i in range(1, 5)]
    arrays = [np.zeros((size, size), dtype=np.uint8) for size in sizes]
    for arr in arrays:
        arr[:10, :10] = 255
        arr[-10:, -10:] = 255

    viewer.add_image(arrays, multiscale=True, rotate=44)
    screenshot_rgba = viewer.screenshot(canvas_only=True, flash=False)
    screenshot_rgb = screenshot_rgba[..., :3]
    assert np.any(
        screenshot_rgb
    )  # make sure there is at least one white pixel
