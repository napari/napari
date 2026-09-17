import numpy as np
from qtpy.QtCore import Qt

from napari._vispy.utils.visual import get_view_direction_in_scene_coordinates


def test_get_view_direction_in_scene_coordinates(make_napari_viewer):
    viewer = make_napari_viewer()

    # Note: as of 0.5.6, setting the dims ndim to 3 with no layers leaves the
    # viewer in an inconsistent state, because the dims are 3 but the layers
    # extent is only 2D. Therefore, instead of setting dims to 3 we add a 3D
    # dataset to the viewer
    _ = viewer.add_image(np.random.random((2, 3, 4)))

    # reset view sets the camera angles to (0, 0, 90)
    viewer.dims.ndisplay = 3

    # get the viewbox
    view_box = viewer.window._qt_viewer.canvas.view

    # get the view direction
    view_dir = get_view_direction_in_scene_coordinates(
        view_box, viewer.dims.ndim, viewer.dims.displayed
    )
    np.testing.assert_allclose(view_dir, [-1, 0, 0], atol=1e-8)


def test_get_view_direction_in_scene_coordinates_2d(make_napari_viewer):
    """view_direction should be None in 2D"""
    viewer = make_napari_viewer()

    # reset view sets the camera angles to (0, 0, 90)
    viewer.dims.ndim = 3
    viewer.dims.ndisplay = 2

    # get the viewbox
    view_box = viewer.window._qt_viewer.canvas.view

    # get the view direction
    view_dir = get_view_direction_in_scene_coordinates(
        view_box, viewer.dims.ndim, viewer.dims.displayed
    )

    assert view_dir is None


def test_set_cursor(make_napari_viewer):
    viewer = make_napari_viewer()
    labels_layer = viewer.add_labels(np.zeros((10, 10), dtype=int))
    brush_overlay = labels_layer._overlays['brush_circle']

    # The labels layer uses the standard cursor by default
    assert (
        viewer.window._qt_viewer.canvas.cursor.shape()
        == Qt.CursorShape.ArrowCursor
    )
    assert not brush_overlay.visible

    # use a known zoom so that brush size equals cursor size
    viewer.scene.camera.zoom = 1

    # Paint mode uses a blank cursor and shows the brush circle overlay
    labels_layer.mode = 'paint'
    assert brush_overlay.visible
    assert (
        viewer.window._qt_viewer.canvas.cursor.shape()
        == Qt.CursorShape.BlankCursor
    )

    # A brush that is too small falls back to the standard cursor
    labels_layer.brush_size = 0
    assert (
        viewer.window._qt_viewer.canvas.cursor.shape()
        == Qt.CursorShape.ArrowCursor
    )
    assert not brush_overlay.visible

    # A normal sized brush shows the blank cursor and the overlay again
    labels_layer.brush_size = 20
    assert brush_overlay.visible
    assert (
        viewer.window._qt_viewer.canvas.cursor.shape()
        == Qt.CursorShape.BlankCursor
    )

    # A brush that is larger than the canvas falls back to the standard cursor
    viewer.scene.camera.zoom = 100
    assert not brush_overlay.visible
    assert (
        viewer.window._qt_viewer.canvas.cursor.shape()
        == Qt.CursorShape.ArrowCursor
    )
