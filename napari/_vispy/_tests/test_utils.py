import numpy as np
import pytest
from qtpy.QtCore import Qt
from vispy.util.quaternion import Quaternion

from napari._pydantic_compat import ValidationError
from napari._vispy.utils.cursor import QtCursorVisual
from napari._vispy.utils.quaternion import quaternion2euler_degrees
from napari._vispy.utils.visual import get_view_direction_in_scene_coordinates
from napari.components._viewer_constants import CursorStyle

# Euler angles to be tested, in degrees
angles = [[12, 53, 92], [180, -90, 0], [16, 90, 0]]

# Prepare for input and add corresponding values in radians


@pytest.mark.parametrize('angles', angles)
def test_quaternion2euler_degrees(angles):
    """Test quaternion to euler angle conversion."""

    # Test for degrees
    q = Quaternion.create_from_euler_angles(*angles, degrees=True)
    ea = quaternion2euler_degrees(q)
    q_p = Quaternion.create_from_euler_angles(*ea, degrees=True)

    # We now compare the corresponding quaternions ; they should be equals or opposites (as they're already unit ones)
    q_values = np.array([q.w, q.x, q.y, q.z])
    q_p_values = np.array([q_p.w, q_p.x, q_p.y, q_p.z])

    nn_zero_ind = np.argmax((q_values != 0) & (q_p_values != 0))

    q_values *= np.sign(q_values[nn_zero_ind])
    q_p_values *= np.sign(q_p_values[nn_zero_ind])

    np.testing.assert_allclose(q_values, q_p_values)


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
    np.testing.assert_allclose(view_dir, [1, 0, 0], atol=1e-8)


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
    viewer.cursor.style = CursorStyle.SQUARE.value
    viewer.cursor.size = 10
    assert (
        viewer.window._qt_viewer.canvas.cursor.shape()
        == Qt.CursorShape.BitmapCursor
    )

    viewer.cursor.size = 5
    assert (
        viewer.window._qt_viewer.canvas.cursor.shape()
        == QtCursorVisual['cross'].value
    )

    viewer.cursor.style = CursorStyle.CIRCLE.value
    viewer.cursor.size = 100

    assert viewer._brush_circle_overlay.visible
    assert viewer._brush_circle_overlay.size == viewer.cursor.size

    with pytest.raises(ValidationError):
        viewer.cursor.style = 'invalid'
