import collections
import os
import sys

import numpy as np
import pytest

from napari.utils.interactions import (
    ReadOnlyWrapper,
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)

skip_on_win_ci = pytest.mark.skipif(
    sys.platform.startswith('win') and os.getenv('CI', '0') != '0',
    reason='Screenshot tests are not supported on windows CI.',
)
skip_local_popups = pytest.mark.skipif(
    not os.getenv('CI') and os.getenv('NAPARI_POPUP_TESTS', '0') == '0',
    reason='Tests requiring GUI windows are skipped locally by default.',
)


@skip_on_win_ci
@skip_local_popups
def test_z_order_adding_removing_images(make_napari_viewer):
    """Test z order is correct after adding/ removing images."""
    data = np.ones((10, 10))

    viewer = make_napari_viewer(show=True)
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


@skip_on_win_ci
@skip_local_popups
def test_z_order_images(make_napari_viewer):
    """Test changing order of images changes z order in display."""
    data = np.ones((10, 10))

    viewer = make_napari_viewer(show=True)
    viewer.add_image(data, colormap='red')
    viewer.add_image(data, colormap='blue')
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    viewer.layers.move(1, 0)
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that red is now visible
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])


@skip_on_win_ci
@skip_local_popups
def test_z_order_image_points(make_napari_viewer):
    """Test changing order of image and points changes z order in display."""
    data = np.ones((10, 10))

    viewer = make_napari_viewer(show=True)
    viewer.add_image(data, colormap='red')
    viewer.add_points([5, 5], face_color='blue', size=10)
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that blue is visible
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    viewer.layers.move(1, 0)
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    # Check that red is now visible
    np.testing.assert_almost_equal(screenshot[center], [255, 0, 0, 255])


@skip_on_win_ci
@skip_local_popups
def test_z_order_images_after_ndisplay(make_napari_viewer):
    """Test z order of images remanins constant after chaning ndisplay."""
    data = np.ones((10, 10))

    viewer = make_napari_viewer(show=True)
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


@skip_on_win_ci
@skip_local_popups
def test_z_order_image_points_after_ndisplay(make_napari_viewer):
    """Test z order of image and points remanins constant after chaning ndisplay."""
    data = np.ones((10, 10))

    viewer = make_napari_viewer(show=True)
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


@skip_on_win_ci
@skip_local_popups
def test_changing_image_colormap(make_napari_viewer):
    """Test changing colormap changes rendering."""
    viewer = make_napari_viewer(show=True)

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


@skip_on_win_ci
@skip_local_popups
def test_changing_image_gamma(make_napari_viewer):
    """Test changing gamma changes rendering."""
    viewer = make_napari_viewer(show=True)

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


@skip_on_win_ci
@skip_local_popups
def test_grid_mode(make_napari_viewer):
    """Test changing gamma changes rendering."""
    viewer = make_napari_viewer(show=True)

    # Add images
    data = np.ones((6, 15, 15))
    viewer.add_image(data, channel_axis=0, blending='translucent')

    assert not viewer.grid.enabled
    assert viewer.grid.actual_shape(6) == (1, 1)
    assert viewer.grid.stride == 1
    translations = [layer.translate_grid for layer in viewer.layers]
    expected_translations = np.zeros((6, 2))
    np.testing.assert_allclose(translations, expected_translations)

    # check screenshot
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    np.testing.assert_almost_equal(screenshot[center], [0, 0, 255, 255])

    # enter grid view
    viewer.grid.enabled = True
    assert viewer.grid.enabled
    assert viewer.grid.actual_shape(6) == (2, 3)
    assert viewer.grid.stride == 1
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
        (2 / 3, 1 / 3),
        (2 / 3, 1 / 2),
        (2 / 3, 2 / 3),
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

    # reorder layers, swapping 0 and 5
    viewer.layers.move(5, 0)
    viewer.layers.move(1, 6)

    # check screenshot
    screenshot = viewer.screenshot(canvas_only=True)
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

    # return to stack view
    viewer.grid.enabled = False
    assert not viewer.grid.enabled
    assert viewer.grid.actual_shape(6) == (1, 1)
    assert viewer.grid.stride == 1
    translations = [layer.translate_grid for layer in viewer.layers]
    expected_translations = np.zeros((6, 2))
    np.testing.assert_allclose(translations, expected_translations)

    # check screenshot
    screenshot = viewer.screenshot(canvas_only=True)
    center = tuple(np.round(np.divide(screenshot.shape[:2], 2)).astype(int))
    np.testing.assert_almost_equal(screenshot[center], [0, 255, 255, 255])


@skip_on_win_ci
@skip_local_popups
def test_changing_image_attenuation(make_napari_viewer):
    """Test changing attenuation value changes rendering."""
    data = np.zeros((100, 10, 10))
    data[-1] = 1

    viewer = make_napari_viewer(show=True)
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


@skip_on_win_ci
@skip_local_popups
def test_labels_painting(make_napari_viewer):
    """Test painting labels updates image."""
    data = np.zeros((100, 100), dtype=np.int32)

    viewer = make_napari_viewer(show=True)
    viewer.add_labels(data)
    layer = viewer.layers[0]

    screenshot = viewer.screenshot(canvas_only=True)

    # Check that no painting has occurred
    assert layer.data.max() == 0
    assert screenshot[:, :, :2].max() == 0

    # Enter paint mode
    viewer.cursor.position = (0, 0)
    layer.mode = 'paint'
    layer.selected_label = 3

    # Simulate click
    Event = collections.namedtuple(
        'Event', field_names=['type', 'is_dragging', 'position']
    )

    # Simulate click
    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            position=viewer.cursor.position,
        )
    )
    mouse_press_callbacks(layer, event)

    viewer.cursor.position = (100, 100)

    # Simulate drag
    event = ReadOnlyWrapper(
        Event(
            type='mouse_move',
            is_dragging=True,
            position=viewer.cursor.position,
        )
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = ReadOnlyWrapper(
        Event(
            type='mouse_release',
            is_dragging=False,
            position=viewer.cursor.position,
        )
    )
    mouse_release_callbacks(layer, event)

    event = ReadOnlyWrapper(
        Event(
            type='mouse_press',
            is_dragging=False,
            position=viewer.cursor.position,
        )
    )
    mouse_press_callbacks(layer, event)

    screenshot = viewer.screenshot(canvas_only=True)
    # Check that painting has now occurred
    assert layer.data.max() > 0
    assert screenshot[:, :, :2].max() > 0


@pytest.mark.skip("Welcome visual temporarily disabled")
@skip_on_win_ci
@skip_local_popups
def test_welcome(make_napari_viewer):
    """Test that something visible on launch."""
    viewer = make_napari_viewer(show=True)

    # Check something is visible
    screenshot = viewer.screenshot(canvas_only=True)
    assert len(viewer.layers) == 0
    assert screenshot[..., :-1].max() > 0

    # Check adding zeros image makes it go away
    viewer.add_image(np.zeros((1, 1)))
    screenshot = viewer.screenshot(canvas_only=True)
    assert len(viewer.layers) == 1
    assert screenshot[..., :-1].max() == 0

    # Remove layer and check something is visible again
    viewer.layers.pop(0)
    screenshot = viewer.screenshot(canvas_only=True)
    assert len(viewer.layers) == 0
    assert screenshot[..., :-1].max() > 0


@skip_on_win_ci
@skip_local_popups
def test_axes_visible(make_napari_viewer):
    """Test that something appears when axes become visible."""
    viewer = make_napari_viewer(show=True)
    viewer.window.qt_viewer.set_welcome_visible(False)

    # Check axes are not visible
    launch_screenshot = viewer.screenshot(canvas_only=True)
    assert not viewer.axes.visible

    # Make axes visible and check something is seen
    viewer.axes.visible = True
    on_screenshot = viewer.screenshot(canvas_only=True)
    assert viewer.axes.visible
    assert abs(on_screenshot - launch_screenshot).max() > 0

    # Make axes not visible and check they are gone
    viewer.axes.visible = False
    off_screenshot = viewer.screenshot(canvas_only=True)
    assert not viewer.axes.visible
    np.testing.assert_almost_equal(launch_screenshot, off_screenshot)


@skip_on_win_ci
@skip_local_popups
def test_scale_bar_visible(make_napari_viewer):
    """Test that something appears when scale bar becomes visible."""
    viewer = make_napari_viewer(show=True)
    viewer.window.qt_viewer.set_welcome_visible(False)

    # Check scale bar is not visible
    launch_screenshot = viewer.screenshot(canvas_only=True)
    assert not viewer.scale_bar.visible

    # Make scale bar visible and check something is seen
    viewer.scale_bar.visible = True
    on_screenshot = viewer.screenshot(canvas_only=True)
    assert viewer.scale_bar.visible
    assert abs(on_screenshot - launch_screenshot).max() > 0

    # Make scale bar not visible and check it is gone
    viewer.scale_bar.visible = False
    off_screenshot = viewer.screenshot(canvas_only=True)
    assert not viewer.scale_bar.visible
    np.testing.assert_almost_equal(launch_screenshot, off_screenshot)
