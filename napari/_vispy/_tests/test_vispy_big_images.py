import numpy as np


def test_big_2D_image(viewer_factory):
    """Test big 2D image with axis exceeding max texture size."""
    view, viewer = viewer_factory()

    shape = (20_000, 10)
    data = np.random.random(shape)
    layer = viewer.add_image(data, is_pyramid=False)
    visual = view.layer_to_visual[layer]
    assert visual.node is not None
    if visual.MAX_TEXTURE_SIZE_2D is not None:
        s = np.ceil(np.divide(shape, visual.MAX_TEXTURE_SIZE_2D)).astype(int)
        assert np.all(layer._transforms['tile2data'].scale == s)


def test_really_big_2d_image(viewer_factory):
    view, viewer = viewer_factory(show=True)

    shape = (6898, 9946)
    data = np.ones(shape)
    _ = viewer.add_image(data)
    screenshot = viewer.screenshot()

    # Screenshot pixel coordinates to test (in format: row, column)
    center_coord = np.round(np.array(screenshot.shape[:2]) / 2).astype(np.int)
    top_left = [40, 36]
    top_right = [40, 763]
    bottom_left = [screenshot.shape[0] - 40, 36]
    bottom_right = [screenshot.shape[0] - 40, 763]

    expected_value = np.array([255, 255, 255, 255])  # white pixel value
    assert all(screenshot[center_coord[0], center_coord[1]] == expected_value)
    assert all(screenshot[top_left[0], top_left[1]] == expected_value)
    assert all(screenshot[top_right[0], top_right[1]] == expected_value)
    assert all(screenshot[bottom_left[0], bottom_left[1]] == expected_value)
    assert all(screenshot[bottom_right[0], bottom_right[1]] == expected_value)


def test_big_3D_image(viewer_factory):
    """Test big 3D image with axis exceeding max texture size."""
    view, viewer = viewer_factory(ndisplay=3)

    shape = (5, 10, 3_000)
    data = np.random.random(shape)
    layer = viewer.add_image(data, is_pyramid=False)
    visual = view.layer_to_visual[layer]
    assert visual.node is not None
    if visual.MAX_TEXTURE_SIZE_3D is not None:
        s = np.ceil(np.divide(shape, visual.MAX_TEXTURE_SIZE_3D)).astype(int)
        assert np.all(layer._transforms['tile2data'].scale == s)
