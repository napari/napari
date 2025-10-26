from __future__ import annotations

import gc
import os
import typing
from itertools import product, takewhile
from math import isclose
from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest
from imageio import imread
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QApplication, QMessageBox
from scipy import ndimage as ndi
from vispy.app import MouseEvent

from napari._qt.qt_viewer import QtViewer
from napari._tests.utils import (
    add_layer_by_type,
    check_viewer_functioning,
    layer_test_data,
    skip_local_popups,
    skip_on_win_ci,
)
from napari._vispy._tests.utils import vispy_image_scene_size
from napari.components.viewer_model import ViewerModel
from napari.layers import Labels, Layer, Points
from napari.settings import get_settings
from napari.utils.colormaps import DirectLabelColormap, label_colormap
from napari.utils.interactions import mouse_press_callbacks

if typing.TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike
    from pytestqt.qtbot import QtBot

BUILTINS_DISP = 'napari'
BUILTINS_NAME = 'builtins'
NUMPY_INTEGER_TYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]


@pytest.fixture
def qt_viewer(
    qtbot: QtBot, qt_viewer_: QtViewer, request: pytest.FixtureRequest
) -> QtViewer:
    if 'show_qt_viewer' in request.keywords:
        qt_viewer_.show()
        qt_viewer_.resize(460, 460)
        QApplication.processEvents()
        qtbot.wait_exposed(qt_viewer_)
    return qt_viewer_


@pytest.mark.parametrize(('layer_class', 'data', 'ndim'), layer_test_data)
def test_add_layer(
    qt_viewer: QtViewer,
    viewer_model: ViewerModel,
    layer_class: type[Layer],
    data: ArrayLike,
    ndim: int,
) -> None:
    viewer_model.dims.ndisplay = np.clip(ndim, 2, 3)

    add_layer_by_type(viewer_model, layer_class, data)
    check_viewer_functioning(viewer_model, qt_viewer, data, ndim)


def test_new_labels(qt_viewer: QtViewer, viewer_model: ViewerModel) -> None:
    """Test adding new labels layer to empty viewer."""
    viewer_model._new_labels()
    assert np.max(viewer_model.layers[0].data) == 0
    assert len(viewer_model.layers) == 1
    assert qt_viewer.layers.model().rowCount() == len(viewer_model.layers)

    assert viewer_model.dims.ndim == 2
    assert qt_viewer.dims.nsliders == viewer_model.dims.ndim
    npt.assert_array_equal(qt_viewer.dims._displayed_sliders, False)


def test_new_labels_to_image(
    qt_viewer: QtViewer, viewer_model: ViewerModel
) -> None:
    """Test adding new labels layer to viewer with image."""
    data = np.random.default_rng(0).random((10, 15))
    viewer_model.add_image(data)
    viewer_model._new_labels()
    assert np.max(viewer_model.layers[1].data) == 0
    assert len(viewer_model.layers) == 2
    assert qt_viewer.layers.model().rowCount() == len(viewer_model.layers)

    assert viewer_model.dims.ndim == 2
    assert qt_viewer.dims.nsliders == viewer_model.dims.ndim
    npt.assert_array_equal(qt_viewer.dims._displayed_sliders, False)


def test_new_points(qt_viewer: QtViewer, viewer_model: ViewerModel) -> None:
    """Test adding a new points layer to empty viewer."""
    viewer_model.add_points()
    assert len(viewer_model.layers[0].data) == 0
    assert len(viewer_model.layers) == 1
    assert qt_viewer.layers.model().rowCount() == len(viewer_model.layers)

    assert viewer_model.dims.ndim == 2
    assert qt_viewer.dims.nsliders == viewer_model.dims.ndim
    npt.assert_array_equal(qt_viewer.dims._displayed_sliders, False)


def test_new_points_to_image(
    qt_viewer: QtViewer, viewer_model: ViewerModel
) -> None:
    """Test adding new points layer to viewer with image."""
    data = np.random.default_rng(0).random((10, 15))
    viewer_model.add_image(data)
    viewer_model.add_points()
    assert len(viewer_model.layers[1].data) == 0
    assert len(viewer_model.layers) == 2
    assert qt_viewer.layers.model().rowCount() == len(viewer_model.layers)

    assert viewer_model.dims.ndim == 2
    assert qt_viewer.dims.nsliders == viewer_model.dims.ndim
    npt.assert_array_equal(qt_viewer.dims._displayed_sliders, False)


def test_new_shapes_empty_viewer(
    qt_viewer: QtViewer, viewer_model: ViewerModel
) -> None:
    """Test adding new shapes layer to empty viewer."""
    viewer_model.add_shapes()
    assert len(viewer_model.layers[0].data) == 0
    assert len(viewer_model.layers) == 1
    assert qt_viewer.layers.model().rowCount() == len(viewer_model.layers)

    assert viewer_model.dims.ndim == 2
    assert qt_viewer.dims.nsliders == viewer_model.dims.ndim
    npt.assert_array_equal(qt_viewer.dims._displayed_sliders, False)


def test_new_shapes_to_image(
    qt_viewer: QtViewer, viewer_model: ViewerModel
) -> None:
    """Test adding new shapes layer to viewer with image."""
    data = np.random.default_rng(0).random((10, 15))
    viewer_model.add_image(data)
    viewer_model.add_shapes()
    assert len(viewer_model.layers[1].data) == 0
    assert len(viewer_model.layers) == 2
    assert qt_viewer.layers.model().rowCount() == len(viewer_model.layers)

    assert viewer_model.dims.ndim == 2
    assert qt_viewer.dims.nsliders == viewer_model.dims.ndim
    npt.assert_array_equal(qt_viewer.dims._displayed_sliders, False)


def test_z_order_adding_removing_images(
    viewer_model: ViewerModel, qt_viewer: QtViewer
) -> None:
    """Test z order is correct after adding/ removing images."""
    data = np.ones((10, 10))

    vis = qt_viewer.canvas.layer_to_visual
    viewer_model.add_image(data, colormap='red', name='red')
    viewer_model.add_image(data, colormap='green', name='green')
    viewer_model.add_image(data, colormap='blue', name='blue')
    order = [vis[x].order for x in viewer_model.layers]
    np.testing.assert_almost_equal(
        order, list(range(len(viewer_model.layers)))
    )

    # Remove and re-add image
    viewer_model.layers.remove('red')
    order = [vis[x].order for x in viewer_model.layers]
    np.testing.assert_almost_equal(
        order, list(range(len(viewer_model.layers)))
    )
    viewer_model.add_image(data, colormap='red', name='red')
    order = [vis[x].order for x in viewer_model.layers]
    np.testing.assert_almost_equal(
        order, list(range(len(viewer_model.layers)))
    )

    # Remove two other images
    viewer_model.layers.remove('green')
    viewer_model.layers.remove('blue')
    order = [vis[x].order for x in viewer_model.layers]
    np.testing.assert_almost_equal(
        order, list(range(len(viewer_model.layers)))
    )

    # Add two other layers back
    viewer_model.add_image(data, colormap='green', name='green')
    viewer_model.add_image(data, colormap='blue', name='blue')
    order = [vis[x].order for x in viewer_model.layers]
    np.testing.assert_almost_equal(
        order, list(range(len(viewer_model.layers)))
    )


@pytest.mark.show_qt_viewer
def test_export_figure(
    qt_viewer: QtViewer,
    viewer_model: ViewerModel,
    tmp_path: Path,
    qtbot: QtBot,
) -> None:
    # Add image
    data = np.ones((250, 250))
    layer = viewer_model.add_image(data)

    camera_center = viewer_model.camera.center
    camera_zoom = viewer_model.camera.zoom
    img = qt_viewer.export_figure(flash=False, path=str(tmp_path / 'img.png'))

    assert viewer_model.camera.center == camera_center
    assert isclose(viewer_model.camera.zoom, camera_zoom)
    np.testing.assert_allclose(img.shape, (250, 250, 4), atol=1)

    assert (img.reshape(-1, 4) == [255, 255, 255, 255]).all(axis=1).all()

    assert (tmp_path / 'img.png').exists()

    layer.scale = [0.12, 0.24]
    img = qt_viewer.export_figure(flash=False)
    # allclose accounts for rounding errors when computing size in hidpi aka
    # retina displays
    np.testing.assert_allclose(img.shape, (250, 500, 4), atol=1)

    layer.scale = [0.12, 0.12]
    img = qt_viewer.export_figure(flash=False)
    np.testing.assert_allclose(img.shape, (250, 250, 4), atol=1)


@pytest.mark.show_qt_viewer
def test_export_figure_3d(
    qt_viewer: QtViewer,
    viewer_model: ViewerModel,
    tmp_path: Path,
    qtbot: QtBot,
) -> None:
    rng = np.random.default_rng(0)
    # Add image, keep values low to contrast with white background
    viewer_model.dims.ndisplay = 3
    viewer_model.theme = 'light'

    data = rng.integers(50, 100, size=(10, 250, 250), dtype=np.uint8)
    layer = viewer_model.add_image(data)

    # check the non-rotated data (angles = 0,0,90) are exported without any
    # visible background, since the margins should be 0
    img = qt_viewer.export_figure(flash=False)
    np.testing.assert_allclose(img.shape, (250, 250, 4), atol=1)

    # check that changing the scale still gives the pixel size
    layer.scale = [1, 0.12, 0.24]
    img = qt_viewer.export_figure(flash=False)
    np.testing.assert_allclose(img.shape, (250, 500, 4), atol=1)
    layer.scale = [1, 1, 1]

    # rotate the data, export the figure, and check that the rotated figure
    # shape is greater than the original data shape
    viewer_model.camera.angles = (45, 45, 45)
    img = qt_viewer.export_figure(flash=False)
    np.testing.assert_allclose(img.shape, (171, 339, 4), atol=1)

    # FIXME: Changes introduced in #7870 slightly changed the timing and result in a blank canvas.
    # Probably related to #8033. Because canvass size is still correct, we know it would look alright
    # The theme is dark, so the canvas will be white. Test that the image
    # has a white background, roughly more background than the data itself.
    # assert (img[img > 250].shape[0] / img[img <= 200].shape[0]) > 0.5


@pytest.mark.show_qt_viewer
def test_export_rois(
    qt_viewer: QtViewer,
    viewer_model: ViewerModel,
    tmp_path: Path,
    qtbot: QtBot,
) -> None:
    # Create an image with a defined shape (100x100) and a square in the middle

    img = np.zeros((100, 100), dtype=np.uint8)
    img[25:75, 25:75] = 255

    # Add viewer
    viewer_model.add_image(img, colormap='gray')

    # Create a couple of clearly defined rectangular polygons for validation
    roi_shapes_data = [
        np.array([[0, 0], [20, 0], [20, 20], [0, 20]]) - (0.5, 0.5),
        np.array([[15, 15], [35, 15], [35, 35], [15, 35]]) - (0.5, 0.5),
        np.array([[65, 65], [85, 65], [85, 85], [65, 85]]) - (0.5, 0.5),
        np.array([[15, 65], [35, 65], [35, 85], [15, 85]]) - (0.5, 0.5),
        np.array([[65, 15], [85, 15], [85, 35], [65, 35]]) - (0.5, 0.5),
        np.array([[40, 40], [60, 40], [60, 60], [40, 60]]) - (0.5, 0.5),
    ]
    paths = [
        str(tmp_path / f'roi_{i}.png') for i in range(len(roi_shapes_data))
    ]

    # Save original camera state for comparison later
    camera_center = viewer_model.camera.center
    camera_zoom = viewer_model.camera.zoom

    with pytest.raises(ValueError, match='The number of file'):
        qt_viewer.export_rois(roi_shapes_data, paths=paths + ['fake'])
    # Export ROI to image path
    test_roi = qt_viewer.export_rois(roi_shapes_data, paths=paths)
    qtbot.wait(10)

    assert all(
        (tmp_path / f'roi_{i}.png').exists()
        for i in range(len(roi_shapes_data))
    )

    # This test uses scaling to adjust the expected size of ROI images
    # and number of white pixels in the ROI screenshots
    # The assertion may fail if the test is run on screens with fractional scaling.
    scaling = qt_viewer.screen().devicePixelRatio()

    assert all(
        roi.shape == (20 * scaling, 20 * scaling, 4) for roi in test_roi
    )
    assert viewer_model.camera.center == camera_center
    assert viewer_model.camera.zoom == camera_zoom

    test_dir = tmp_path / 'test_dir'
    refs = []
    import gc

    try:
        qtbot.wait(1000)
        res = qt_viewer.export_rois(roi_shapes_data, paths=test_dir)
        refs.append(res)
        gc.collect()
        QApplication.processEvents()
        qtbot.wait(1000)
    finally:
        res.clear()
        gc.collect()

    assert all(
        (test_dir / f'roi_{i}.png').exists()
        for i in range(len(roi_shapes_data))
    )
    expected_values = [0, 100, 100, 100, 100, 400]
    for index, roi_img in enumerate(test_roi):
        gray_img = roi_img[..., 0]
        assert (
            np.count_nonzero(gray_img) == expected_values[index] * scaling**2
        ), f'Wrong number of white pixels in the ROI {index}'


@pytest.mark.show_qt_viewer
def test_export_rois_3d_fail(
    qt_viewer: QtViewer, viewer_model: ViewerModel
) -> None:
    # create 3d ROI for testing
    roi_3d = [
        np.array([[0, 0, 0], [0, 20, 0], [0, 20, 20], [0, 0, 20]]),
        np.array([[0, 15, 15], [0, 35, 15], [0, 35, 35], [0, 15, 35]]),
    ]

    # Only 2D roi supported at the moment
    with pytest.raises(ValueError, match='ROI found with invalid'):
        qt_viewer.export_rois(roi_3d)

    test_data = np.zeros((4, 50, 50))
    viewer_model.add_image(test_data)
    viewer_model.dims.ndisplay = 3

    # 3D view should fail
    roi_data = [
        np.array([[0, 0], [20, 0], [20, 20], [0, 20]]),
        np.array([[15, 15], [35, 15], [35, 35], [15, 35]]),
    ]
    with pytest.raises(
        NotImplementedError, match="'export_rois' is not implemented"
    ):
        qt_viewer.export_rois(roi_data)


@pytest.mark.skip('new approach')
@pytest.mark.show_qt_viewer
def test_screenshot_dialog(
    viewer_model: ViewerModel, qt_viewer: QtViewer, tmp_path: Path
) -> None:
    """Test save screenshot functionality."""
    rng = np.random.default_rng(0)
    # Add image
    data = rng.random((10, 15))
    viewer_model.add_image(data)

    # Add labels
    data = rng.integers(20, size=(10, 15))
    viewer_model.add_labels(data)

    # Add points
    data = 20 * rng.random((10, 2))
    viewer_model.add_points(data)

    # Add vectors
    data = 20 * rng.random((10, 2, 2))
    viewer_model.add_vectors(data)

    # Add shapes
    data = 20 * rng.random((10, 4, 2))
    viewer_model.add_shapes(data)

    # Save screenshot
    input_filepath = os.path.join(tmp_path, 'test-save-screenshot')
    mock_return = (input_filepath, '')
    with (
        mock.patch('napari._qt._qt_viewer.QFileDialog') as mocker,
        mock.patch('napari._qt._qt_viewer.QMessageBox') as mocker2,
    ):
        mocker.getSaveFileName.return_value = mock_return
        mocker2.warning.return_value = QMessageBox.Yes
        qt_viewer._screenshot_dialog()

    # Assert behaviour is correct
    expected_filepath = input_filepath + '.png'  # add default file extension
    assert os.path.exists(expected_filepath)
    output_data = imread(expected_filepath)
    expected_data = qt_viewer.screenshot(flash=False)
    assert np.allclose(output_data, expected_data)


@pytest.mark.key_bindings
def test_active_keybindings(
    qt_viewer: QtViewer, viewer_model: ViewerModel
) -> None:
    """Test instantiating viewer."""
    # Check only keybinding is Viewer
    assert len(qt_viewer._key_map_handler.keymap_providers) == 1
    assert qt_viewer._key_map_handler.keymap_providers[0] == viewer_model

    # Add a layer and check it is keybindings are active
    data = np.random.default_rng(0).random((10, 15))
    layer_image = viewer_model.add_image(data)
    assert viewer_model.layers.selection.active == layer_image
    assert len(qt_viewer._key_map_handler.keymap_providers) == 2
    assert qt_viewer._key_map_handler.keymap_providers[0] == layer_image

    # Add a layer and check it is keybindings become active
    layer_image_2 = viewer_model.add_image(data)
    assert viewer_model.layers.selection.active == layer_image_2
    assert len(qt_viewer._key_map_handler.keymap_providers) == 2
    assert qt_viewer._key_map_handler.keymap_providers[0] == layer_image_2

    # Change active layer and check it is keybindings become active
    viewer_model.layers.selection.active = layer_image
    assert viewer_model.layers.selection.active == layer_image
    assert len(qt_viewer._key_map_handler.keymap_providers) == 2
    assert qt_viewer._key_map_handler.keymap_providers[0] == layer_image


def test_process_mouse_event(
    qt_viewer: QtViewer, viewer_model: ViewerModel
) -> None:
    """Test that the correct properties are added to the
    MouseEvent by _process_mouse_events.
    """
    # make a mock mouse event
    new_pos = (25, 25)
    mouse_event = MouseEvent(
        type='mouse_press',
        pos=new_pos,
    )
    data = np.zeros((5, 20, 20, 20), dtype=int)
    data[1, 0:10, 0:10, 0:10] = 1

    labels = viewer_model.add_labels(
        data, scale=(1, 2, 1, 1), translate=(5, 5, 5)
    )

    @labels.mouse_drag_callbacks.append
    def on_click(layer, event):
        np.testing.assert_almost_equal(event.view_direction, [0, -1, 0, 0])
        np.testing.assert_array_equal(event.dims_displayed, [1, 2, 3])
        assert event.dims_point[0] == data.shape[0] // 2

        expected_position = qt_viewer.canvas._map_canvas2world(
            new_pos, qt_viewer.canvas.view
        )
        np.testing.assert_almost_equal(expected_position, list(event.position))

    viewer_model.dims.ndisplay = 3
    qt_viewer.canvas._process_mouse_event(mouse_press_callbacks, mouse_event)


def test_process_mouse_event_2d_layer_3d_viewer(
    qt_viewer: QtViewer, viewer_model: ViewerModel
) -> None:
    """Test that _process_mouse_events can handle 2d layers in 3D.

    This is a test for: https://github.com/napari/napari/issues/7299
    """

    # make a mock mouse event
    new_pos = (5, 5)
    mouse_event = MouseEvent(
        type='mouse_press',
        pos=new_pos,
    )
    data = np.zeros((20, 20))

    image = viewer_model.add_image(data)

    @image.mouse_drag_callbacks.append
    def on_click(layer, event):
        expected_position = qt_viewer.canvas._map_canvas2world(
            new_pos, qt_viewer.canvas.view
        )
        np.testing.assert_almost_equal(expected_position, list(event.position))

    assert viewer_model.dims.ndisplay == 2
    qt_viewer.canvas._process_mouse_event(mouse_press_callbacks, mouse_event)

    viewer_model.dims.ndisplay = 3
    qt_viewer.canvas._process_mouse_event(mouse_press_callbacks, mouse_event)


@pytest.mark.usefixtures(
    'qt_viewer'
)  # need qt_viewer to trigger the vispy code
def test_remove_points(viewer_model: ViewerModel) -> None:
    viewer_model.add_points([(1, 2), (2, 3)])
    del viewer_model.layers[0]
    viewer_model.add_points([(1, 2), (2, 3)])


@pytest.mark.usefixtures(
    'qt_viewer'
)  # need qt_viewer to trigger the vispy code
def test_remove_image(viewer_model: ViewerModel) -> None:
    rng = np.random.default_rng(0)
    viewer_model.add_image(rng.random((10, 10)))
    del viewer_model.layers[0]
    viewer_model.add_image(rng.random((10, 10)))


@pytest.mark.usefixtures(
    'qt_viewer'
)  # need qt_viewer to trigger the vispy code
def test_remove_labels(viewer_model: ViewerModel) -> None:
    rng = np.random.default_rng(0)
    viewer_model.add_labels(rng.integers(0, 10, size=(10, 10), dtype=np.int8))
    del viewer_model.layers[0]
    viewer_model.add_labels(rng.integers(0, 10, size=(10, 10), dtype=np.int8))


@pytest.mark.skip(
    reason='Broadcasting layers is broken by reordering dims, see #3882'
)
@pytest.mark.parametrize('multiscale', [False, True])
def test_mixed_2d_and_3d_layers(
    viewer_model: ViewerModel, qt_viewer: QtViewer, multiscale: bool
) -> None:
    """Test bug in setting corner_pixels from qt_viewer.on_draw"""
    img = np.ones((512, 256))
    # canvas size must be large enough that img fits in the canvas
    canvas_size = tuple(3 * s for s in img.shape)
    expected_corner_pixels = np.asarray([[0, 0], [s - 1 for s in img.shape]])

    vol = np.stack([img] * 8, axis=0)
    if multiscale:
        img = [img[::s, ::s] for s in (1, 2, 4)]
    viewer_model.add_image(img)
    img_multi_layer = viewer_model.layers[0]
    viewer_model.add_image(vol)

    viewer_model.dims.order = (0, 1, 2)
    qt_viewer.canvas.size = canvas_size
    qt_viewer.canvas.on_draw(None)
    np.testing.assert_array_equal(
        img_multi_layer.corner_pixels, expected_corner_pixels
    )

    viewer_model.dims.order = (2, 0, 1)
    qt_viewer.canvas.on_draw(None)
    np.testing.assert_array_equal(
        img_multi_layer.corner_pixels, expected_corner_pixels
    )

    viewer_model.dims.order = (1, 2, 0)
    qt_viewer.canvas.on_draw(None)
    np.testing.assert_array_equal(
        img_multi_layer.corner_pixels, expected_corner_pixels
    )


@pytest.mark.usefixtures(
    'qt_viewer'
)  # need qt_viewer to trigger the vispy code
def test_remove_add_image_3D(viewer_model: ViewerModel) -> None:
    """
    Test that adding, removing and readding an image layer in 3D does not cause issues
    due to the vispy node change. See https://github.com/napari/napari/pull/3670
    """
    viewer_model.dims.ndisplay = 3
    img = np.ones((10, 10, 10))

    layer = viewer_model.add_image(img)
    viewer_model.layers.remove(layer)
    viewer_model.layers.append(layer)


@skip_on_win_ci
@skip_local_popups
@pytest.mark.show_qt_viewer
def test_qt_viewer_multscale_image_out_of_view(viewer_model):
    """Test out-of-view multiscale image viewing fix.

    Just verifies that no RuntimeError is raised in this scenario.

    see: https://github.com/napari/napari/issues/3863.
    """
    # show=True required to test fix for OpenGL error
    viewer_model.dims.ndisplay = 2
    viewer_model.add_shapes(
        data=[
            np.array(
                [[1500, 4500], [4500, 4500], [4500, 1500], [1500, 1500]],
                dtype=float,
            )
        ],
        shape_type=['polygon'],
    )
    viewer_model.add_image([np.eye(1024), np.eye(512), np.eye(256)])


def test_insert_layer_ordering(
    viewer_model: ViewerModel, qt_viewer: QtViewer
) -> None:
    """make sure layer ordering is correct in vispy when inserting layers"""
    pl1 = Points()
    pl2 = Points()

    viewer_model.layers.append(pl1)
    viewer_model.layers.insert(0, pl2)

    pl1_vispy = qt_viewer.canvas.layer_to_visual[pl1].node
    pl2_vispy = qt_viewer.canvas.layer_to_visual[pl2].node
    assert pl1_vispy.order == 1
    assert pl2_vispy.order == 0


def test_create_non_empty_viewer_model(qtbot: QtBot) -> None:
    viewer_model = ViewerModel()
    viewer_model.add_points([(1, 2), (2, 3)])

    viewer = QtViewer(viewer=viewer_model)

    viewer.close()
    viewer.deleteLater()
    # try to del local reference for gc.
    del viewer_model
    del viewer
    qtbot.wait(50)
    gc.collect()


def _update_data(
    layer: Labels,
    label: int,
    qtbot: QtBot,
    qt_viewer: QtViewer,
    dtype: np.dtype = np.uint64,
) -> tuple[np.ndarray, np.ndarray]:
    """Change layer data and return color of label and middle pixel of screenshot."""
    layer.data = np.full((2, 2), label, dtype=dtype)
    layer.selected_label = label

    qtbot.wait(50)  # wait for .update() to be called on QtColorBox from Qt

    color_box_color = qt_viewer.controls.widgets[
        layer
    ]._label_control.colorbox.color
    screenshot = qt_viewer.screenshot(flash=False)
    shape = np.array(screenshot.shape[:2])
    middle_pixel = screenshot[tuple(shape // 2)]

    return color_box_color, middle_pixel


@pytest.fixture
def qt_viewer_with_controls(qt_viewer: QtViewer) -> QtViewer:
    qt_viewer.controls.show()
    return qt_viewer


@skip_local_popups
@skip_on_win_ci
@pytest.mark.parametrize(
    'use_selection', [True, False], ids=['selected', 'all']
)
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int64])
@pytest.mark.show_qt_viewer
def test_label_colors_matching_widget_auto(
    qtbot: QtBot,
    qt_viewer_with_controls: QtViewer,
    use_selection: bool,
    dtype: np.dtype,
) -> None:
    """Make sure the rendered label colors match the QtColorBox widget."""

    rng = np.random.default_rng(0)
    data = np.ones((2, 2), dtype=dtype)
    layer = qt_viewer_with_controls.viewer.add_labels(data)
    layer.show_selected_label = use_selection
    layer.opacity = 1.0  # QtColorBox & single layer are blending differently
    n_c = len(layer.colormap)

    test_colors = np.concatenate(
        (
            np.arange(1, 10, dtype=dtype),
            [n_c - 1, n_c, n_c + 1],
            rng.integers(
                1, min(2**20, np.iinfo(dtype).max), size=20, dtype=dtype
            ),
            [-1, -2, -10],
        )
    )

    for label in test_colors:
        # Change color & selected color to the same label
        color_box_color, middle_pixel = _update_data(
            layer, label, qtbot, qt_viewer_with_controls, dtype
        )

        npt.assert_allclose(
            color_box_color, middle_pixel, atol=1, err_msg=f'label {label}'
        )
        # there is a difference of rounding between the QtColorBox and the screenshot


@skip_local_popups
@skip_on_win_ci
@pytest.mark.parametrize(
    'use_selection', [True, False], ids=['selected', 'all']
)
@pytest.mark.parametrize('dtype', [np.uint64, np.uint16, np.uint8, np.int16])
@pytest.mark.show_qt_viewer
def test_label_colors_matching_widget_direct(
    qtbot: QtBot,
    qt_viewer_with_controls: QtViewer,
    use_selection: bool,
    dtype: np.dtype,
):
    """Make sure the rendered label colors match the QtColorBox widget."""
    data = np.ones((2, 2), dtype=dtype)

    test_colors = (1, 2, 3, 8, 150, 50)
    color = {
        0: 'transparent',
        1: 'yellow',
        3: 'blue',
        8: 'red',
        150: 'green',
        None: 'white',
    }
    if np.iinfo(dtype).min < 0:
        color[-1] = 'pink'
        color[-2] = 'orange'
        test_colors = test_colors + (-1, -2, -10)

    colormap = DirectLabelColormap(color_dict=color)
    layer = qt_viewer_with_controls.viewer.add_labels(
        data, opacity=1, colormap=colormap
    )
    layer.show_selected_label = use_selection

    color_box_color, middle_pixel = _update_data(
        layer, 0, qtbot, qt_viewer_with_controls, dtype
    )
    assert np.allclose([0, 0, 0, 255], middle_pixel)

    for label in test_colors:
        # Change color & selected color to the same label
        color_box_color, middle_pixel = _update_data(
            layer, label, qtbot, qt_viewer_with_controls, dtype
        )
        npt.assert_almost_equal(
            color_box_color, middle_pixel, err_msg=f'{label=}'
        )
        npt.assert_almost_equal(
            color_box_color,
            colormap.color_dict.get(label, colormap.color_dict[None]) * 255,
            err_msg=f'{label=}',
        )


def test_axis_labels(viewer_model: ViewerModel, qt_viewer: QtViewer) -> None:
    viewer_model.dims.ndisplay = 3
    layer = viewer_model.add_image(np.zeros((2, 2, 2)), scale=(1, 2, 4))

    layer_visual = qt_viewer.layer_to_visual[layer]
    axes_visual = qt_viewer.canvas._overlay_to_visual[
        viewer_model._overlays['axes']
    ][0]

    layer_visual_size = vispy_image_scene_size(layer_visual)
    assert tuple(layer_visual_size) == (8, 4, 2)
    assert tuple(axes_visual.node.text.text) == ('2', '1', '0')


def _find_margin(data: np.ndarray, additional_margin: int) -> tuple[int, int]:
    """
    helper function to determine margins in test_thumbnail_labels
    """

    mid_x, mid_y = data.shape[0] // 2, data.shape[1] // 2
    x_margin = len(
        list(takewhile(lambda x: np.all(x == 0), data[:, mid_y, :3][::-1]))
    )
    y_margin = len(
        list(takewhile(lambda x: np.all(x == 0), data[mid_x, :, :3][::-1]))
    )
    return x_margin + additional_margin, y_margin + additional_margin


# @pytest.mark.xfail(reason="Fails on CI, but not locally")
@skip_local_popups
@pytest.mark.show_qt_viewer
@pytest.mark.parametrize('direct', [True, False], ids=['direct', 'auto'])
def test_thumbnail_labels(
    qtbot: QtBot, direct: bool, qt_viewer: QtViewer, tmp_path: Path
) -> None:
    # Add labels to empty viewer
    layer = qt_viewer.viewer.add_labels(
        np.array([[0, 1], [2, 3]]), opacity=1.0
    )
    if direct:
        layer.colormap = DirectLabelColormap(
            color_dict={
                0: 'red',
                1: 'green',
                2: 'blue',
                3: 'yellow',
                None: 'black',
            }
        )
    else:
        layer.colormap = label_colormap(49)
    qt_viewer.viewer.reset_view()
    qt_viewer.canvas.native.paintGL()
    QApplication.processEvents()
    qtbot.wait(50)

    canvas_screenshot_ = qt_viewer.screenshot(flash=False)

    import imageio

    imageio.imwrite(tmp_path / 'canvas_screenshot_.png', canvas_screenshot_)
    np.savez(tmp_path / 'canvas_screenshot_.npz', canvas_screenshot_)

    # cut off black border
    margin1, margin2 = _find_margin(canvas_screenshot_, 10)
    canvas_screenshot = canvas_screenshot_[margin1:-margin1, margin2:-margin2]
    assert canvas_screenshot.size > 0, (
        f'{canvas_screenshot_.shape}, {margin1=}, {margin2=}'
    )

    thumbnail = layer.thumbnail
    scaled_thumbnail = ndi.zoom(
        thumbnail,
        np.array(canvas_screenshot.shape) / np.array(thumbnail.shape),
        order=0,
        mode='nearest',
    )
    close = np.isclose(canvas_screenshot, scaled_thumbnail)
    problematic_pixels_count = np.sum(~close)
    assert problematic_pixels_count < 0.01 * canvas_screenshot.size


@pytest.mark.show_qt_viewer
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32])
def test_background_color(
    qtbot: QtBot, qt_viewer: QtViewer, dtype: np.dtype
) -> None:
    data = np.zeros((10, 10), dtype=dtype)
    data[5:] = 10
    layer = qt_viewer.viewer.add_labels(data, opacity=1)
    color = layer.colormap.map(10) * 255

    backgrounds = (0, 2, -2)

    for background in backgrounds:
        data[:5] = background
        layer.data = data
        layer.colormap = label_colormap(49, background_value=background)
        qtbot.wait(50)
        canvas_screenshot = qt_viewer.screenshot(flash=False)
        shape = np.array(canvas_screenshot.shape[:2])
        background_pixel = canvas_screenshot[tuple((shape * 0.25).astype(int))]
        color_pixel = canvas_screenshot[tuple((shape * 0.75).astype(int))]
        npt.assert_array_equal(
            background_pixel,
            [0, 0, 0, 255],
            err_msg=f'background {background}',
        )
        npt.assert_array_equal(
            color_pixel, color, err_msg=f'background {background}'
        )


@pytest.mark.show_qt_viewer
def test_rendering_interpolation(
    qtbot: QtBot, qt_viewer: QtViewer, viewer_model: ViewerModel
) -> None:
    data = np.zeros((20, 20, 20), dtype=np.uint8)
    data[1:-1, 1:-1, 1:-1] = 5
    layer = viewer_model.add_labels(data, opacity=1, rendering='translucent')
    layer.selected_label = 5
    viewer_model.dims.ndisplay = 3
    QApplication.processEvents()
    canvas_screenshot = qt_viewer.screenshot(flash=False)
    shape = np.array(canvas_screenshot.shape[:2])
    pixel = canvas_screenshot[tuple((shape * 0.5).astype(int))]
    color = layer.colormap.map(5) * 255
    npt.assert_array_equal(pixel, color)


@pytest.mark.slow
@pytest.mark.show_qt_viewer
@pytest.mark.parametrize('mode', ['direct', 'random'])
def test_selection_collision(
    qt_viewer: QtViewer,
    viewer_model: ViewerModel,
    mode: typing.Literal['direct', 'random'],
) -> None:
    data = np.zeros((10, 10), dtype=np.uint8)
    data[:5] = 10
    data[5:] = 10 + 49
    layer = viewer_model.add_labels(data, opacity=1)
    layer.selected_label = 10
    if mode == 'direct':
        layer.colormap = DirectLabelColormap(
            color_dict={10: 'red', 10 + 49: 'red', None: 'black'}
        )

    for dtype in NUMPY_INTEGER_TYPES:
        layer.data = data.astype(dtype)
        layer.show_selected_label = False
        QApplication.processEvents()
        canvas_screenshot = qt_viewer.screenshot(flash=False)
        shape = np.array(canvas_screenshot.shape[:2])
        pixel_10 = canvas_screenshot[tuple((shape * 0.25).astype(int))]
        pixel_59 = canvas_screenshot[tuple((shape * 0.75).astype(int))]
        npt.assert_array_equal(pixel_10, pixel_59, err_msg=f'{dtype}')
        assert not np.all(pixel_10 == [0, 0, 0, 255]), dtype

        layer.show_selected_label = True

        canvas_screenshot = qt_viewer.screenshot(flash=False)
        shape = np.array(canvas_screenshot.shape[:2])
        pixel_10_2 = canvas_screenshot[tuple((shape * 0.25).astype(int))]
        pixel_59_2 = canvas_screenshot[tuple((shape * 0.75).astype(int))]

        npt.assert_array_equal(pixel_59_2, [0, 0, 0, 255], err_msg=f'{dtype}')
        npt.assert_array_equal(pixel_10_2, pixel_10, err_msg=f'{dtype}')


@pytest.mark.show_qt_viewer
def test_all_supported_dtypes(
    qt_viewer: QtViewer, viewer_model: ViewerModel
) -> None:
    data = np.zeros((10, 10), dtype=np.uint8)
    layer_ = viewer_model.add_labels(data, opacity=1)

    for i, dtype in enumerate(NUMPY_INTEGER_TYPES, start=1):
        data = np.full((10, 10), i, dtype=dtype)
        layer_.data = data
        QApplication.processEvents()
        canvas_screenshot = qt_viewer.screenshot(flash=False)
        midd_pixel = canvas_screenshot[
            tuple(np.array(canvas_screenshot.shape[:2]) // 2)
        ]
        npt.assert_equal(
            midd_pixel, layer_.colormap.map(i) * 255, err_msg=f'{dtype=} {i=}'
        )

    layer_.colormap = DirectLabelColormap(
        color_dict={
            0: 'red',
            1: 'green',
            2: 'blue',
            3: 'yellow',
            4: 'magenta',
            5: 'cyan',
            6: 'white',
            7: 'pink',
            8: 'orange',
            9: 'purple',
            10: 'brown',
            11: 'gray',
            None: 'black',
        }
    )

    for i, dtype in enumerate(NUMPY_INTEGER_TYPES, start=1):
        data = np.full((10, 10), i, dtype=dtype)
        layer_.data = data
        QApplication.processEvents()
        canvas_screenshot = qt_viewer.screenshot(flash=False)
        midd_pixel = canvas_screenshot[
            tuple(np.array(canvas_screenshot.shape[:2]) // 2)
        ]
        npt.assert_equal(
            midd_pixel, layer_.colormap.map(i) * 255, err_msg=f'{dtype} {i}'
        )


@pytest.mark.slow
@pytest.mark.show_qt_viewer
def test_more_than_uint16_colors(
    qt_viewer: QtViewer, viewer_model: ViewerModel
) -> None:
    pytest.importorskip('numba')
    # this test is slow (10s locally)
    data = np.zeros((10, 10), dtype=np.uint32)
    colors = {
        i: (x, y, z, 1)
        for i, (x, y, z) in zip(
            range(256**2 + 20),
            product(np.linspace(0, 1, 256, endpoint=True), repeat=3),
            strict=False,
        )
    }
    colors[None] = (0, 0, 0, 1)
    layer = viewer_model.add_labels(
        data, opacity=1, colormap=DirectLabelColormap(color_dict=colors)
    )
    assert layer._slice.image.view.dtype == np.float32

    for i in [1, 1000, 100000]:
        data = np.full((10, 10), i, dtype=np.uint32)
        layer.data = data
        canvas_screenshot = qt_viewer.screenshot(flash=False)
        midd_pixel = canvas_screenshot[
            tuple(np.array(canvas_screenshot.shape[:2]) // 2)
        ]
        npt.assert_equal(
            midd_pixel, layer.colormap.map(i) * 255, err_msg=f'{i}'
        )


@skip_local_popups
@pytest.mark.show_qt_viewer
def test_scale_bar_colored(
    qt_viewer: QtViewer, viewer_model: ViewerModel, qtbot
) -> None:
    scale_bar = viewer_model.scale_bar

    # Add black image
    data = np.zeros((2, 2))
    viewer_model.add_image(data)

    # Check scale bar is not visible (all the canvas is black - `[0, 0, 0, 255]`)
    def check_all_black():
        screenshot = qt_viewer.screenshot(flash=False)
        assert np.all(screenshot == [0, 0, 0, 255], axis=-1).all()

    qtbot.waitUntil(check_all_black)

    # Check scale bar is visible (canvas has white `[1, 1, 1, 255]` in it)
    def check_white_scale_bar():
        screenshot = qt_viewer.screenshot(flash=False)
        assert not np.all(screenshot == [0, 0, 0, 255], axis=-1).all()
        assert np.all(screenshot == [255, 255, 255, 255], axis=-1).any()

    scale_bar.visible = True
    qtbot.waitUntil(check_white_scale_bar)

    # Check scale bar is colored (canvas has fuchsia `[1, 0, 1, 255]` and not white in it)
    def check_colored_scale_bar():
        screenshot = qt_viewer.screenshot(flash=False)
        assert not np.all(screenshot == [255, 255, 255, 255], axis=-1).any()
        assert np.all(screenshot == [255, 0, 255, 255], axis=-1).any()

    scale_bar.colored = True
    qtbot.waitUntil(check_colored_scale_bar)

    # Check scale bar is still visible but not colored (canvas has white again but not fuchsia in it)
    def check_only_white_scale_bar():
        screenshot = qt_viewer.screenshot(flash=False)
        assert np.all(screenshot == [255, 255, 255, 255], axis=-1).any()
        assert not np.all(screenshot == [255, 0, 255, 255], axis=-1).any()

    scale_bar.colored = False
    qtbot.waitUntil(check_only_white_scale_bar)


@skip_local_popups
@pytest.mark.show_qt_viewer
def test_scale_bar_ticks(
    qt_viewer: QtViewer, viewer_model: ViewerModel, qtbot
) -> None:
    scale_bar = viewer_model.scale_bar

    # Add black image
    data = np.zeros((2, 2))
    viewer_model.add_image(data)

    # Check scale bar is not visible (all the canvas is black - `[0, 0, 0, 255]`)
    def check_all_black():
        screenshot = qt_viewer.screenshot(flash=False)
        assert np.all(screenshot == [0, 0, 0, 255], axis=-1).all()

    qtbot.waitUntil(check_all_black)

    # Check scale bar is visible (canvas has white `[1, 1, 1, 255]` in it)
    def check_white_scale_bar():
        screenshot = qt_viewer.screenshot(flash=False)
        assert not np.all(screenshot == [0, 0, 0, 255], axis=-1).all()
        assert np.all(screenshot == [255, 255, 255, 255], axis=-1).any()

    scale_bar.visible = True
    qtbot.waitUntil(check_white_scale_bar)

    # Check scale bar has ticks active and take screenshot for later comparison
    assert scale_bar.ticks
    screenshot_with_ticks = qt_viewer.screenshot(flash=False)

    # Check scale bar without ticks (still white present but new screenshot differs from ticks one)
    def check_no_ticks_scale_bar():
        screenshot = qt_viewer.screenshot(flash=False)
        assert np.all(screenshot == [255, 255, 255, 255], axis=-1).any()
        npt.assert_raises(
            AssertionError,
            npt.assert_array_equal,
            screenshot,
            screenshot_with_ticks,
        )

    scale_bar.ticks = False
    qtbot.waitUntil(check_no_ticks_scale_bar)

    # Check scale bar again has ticks (still white present and new screenshot corresponds with ticks one)
    def check_ticks_scale_bar():
        screenshot = qt_viewer.screenshot(flash=False)
        assert np.all(screenshot == [255, 255, 255, 255], axis=-1).any()
        npt.assert_array_equal(screenshot, screenshot_with_ticks)

    scale_bar.ticks = True
    qtbot.waitUntil(check_ticks_scale_bar)


@skip_local_popups
@pytest.mark.usefixtures('qt_viewer')
def test_dask_cache():
    initial_dask_cache = get_settings().application.dask.cache

    # check that disabling dask cache setting calls related logic
    with mock.patch(
        'napari._qt.qt_viewer.resize_dask_cache'
    ) as mock_resize_dask_cache:
        get_settings().application.dask.enabled = False
    mock_resize_dask_cache.assert_called_once_with(
        int(int(False) * initial_dask_cache * 1e9)
    )

    # check that enabling dask cache setting calls related logic
    with mock.patch(
        'napari._qt.qt_viewer.resize_dask_cache'
    ) as mock_resize_dask_cache:
        get_settings().application.dask.enabled = True
    mock_resize_dask_cache.assert_called_once_with(
        int(int(True) * initial_dask_cache * 1e9)
    )


@pytest.mark.show_qt_viewer
def test_viewer_drag_to_zoom(
    qt_viewer: QtViewer, viewer_model: ViewerModel, qtbot: QtBot
) -> None:
    """Test drag to zoom mouse binding."""
    canvas = qt_viewer.canvas

    def zoom_callback(event):
        """Mock zoom callback to check zoom box visibility."""
        data_positions = event.value
        assert len(data_positions) == 2, (
            'Zoom event should release two positions'
        )

    viewer_model._zoom_box.events.zoom.connect(zoom_callback)

    # Add an image layer
    data = np.random.default_rng(0).random((10, 20))
    viewer_model.add_image(data)

    assert viewer_model._zoom_box.visible is False, (
        'Zoom box should be hidden initially'
    )
    qtbot.wait(10)
    # Simulate press to start zooming
    canvas._scene_canvas.events.mouse_press(
        pos=(0, 0), modifiers=('Alt',), button=0
    )
    qtbot.wait(10)
    assert viewer_model._zoom_box.visible is True, (
        'Zoom box should be visible after press'
    )

    # Simulate drag to zoom
    canvas._scene_canvas.events.mouse_move(
        pos=(100, 100),
        modifiers=('Alt',),
        button=0,
        press_event=MouseEvent(
            pos=(0, 0), modifiers=('Alt',), button=0, type='mouse_press'
        ),
    )
    qtbot.wait(10)
    assert viewer_model._zoom_box.visible is True, (
        'Zoom box should remain visible during drag'
    )
    assert viewer_model._zoom_box.position == ((0, 0), (100, 100)), (
        'Zoom box canvas positions should match the drag coordinates'
    )

    # Simulate release to finish zooming
    canvas._scene_canvas.events.mouse_release(
        pos=(100, 100), modifiers=('Alt',), button=0
    )
    qtbot.wait(10)
    assert viewer_model._zoom_box.visible is False, (
        'Zoom box should be hidden after release'
    )


@pytest.mark.show_qt_viewer
def test_viewer_drag_to_zoom_with_cancel(
    qt_viewer: QtViewer, viewer_model: ViewerModel, qtbot: QtBot
) -> None:
    """Test drag to zoom mouse binding."""
    canvas = qt_viewer.canvas

    def zoom_callback(event):
        """Mock zoom callback to check zoom box visibility."""
        data_positions = event.value
        assert len(data_positions) == 2, (
            'Zoom event should release two positions'
        )

    viewer_model._zoom_box.events.zoom.connect(zoom_callback)

    # Add an image layer
    data = np.random.default_rng(0).random((10, 20))
    viewer_model.add_image(data)

    assert viewer_model._zoom_box.visible is False, (
        'Zoom box should be hidden initially'
    )
    qtbot.wait(10)
    # Simulate press to start zooming
    canvas._scene_canvas.events.mouse_press(
        pos=(0, 0), modifiers=('Alt',), button=0
    )
    qtbot.wait(10)
    assert viewer_model._zoom_box.visible is True, (
        'Zoom box should be visible after press'
    )

    # Simulate drag to zoom BUT remove modifiers to cancel
    canvas._scene_canvas.events.mouse_move(
        pos=(100, 100),
        modifiers=(),
        button=0,
        press_event=MouseEvent(
            pos=(0, 0), modifiers=('Alt',), button=0, type='mouse_press'
        ),
    )
    qtbot.wait(10)
    assert viewer_model._zoom_box.visible is False, (
        'Zoom box should remain visible during drag'
    )
    assert viewer_model._zoom_box.position == ((0, 0), (0, 0)), (
        'Zoom box canvas positions should match the drag coordinates'
    )
