"""Camera orientation definitions shared by the napari camera model.

The camera orientation is described by three Euler angles (rx, ry, rz) in
degrees, interpreted as right-handed rotations about the three displayed
dimensions: the first angle rotates about the depth axis (dim0, axis -3), the
second about the vertical axis (dim1, axis -2), and the third about the
horizontal axis (dim2, axis -1), applied in that order. Each angle rotates the
camera about the axis of the corresponding dimension, so a single nonzero
angle rotates the camera about that dimension (e.g. ``(10, 0, 0)`` is a
rotation of 10 degrees about the 0th dimension). Together with an
``orientation`` naming which way the depth, vertical, and horizontal axes
point on the canvas.

With all angles zero the camera shows the "home view": the view direction lies
along the depth axis, up along the vertical axis, and right along the
horizontal axis, with signs following the orientation. The math here is pure
and backend-agnostic; renderers (e.g. VisPy) are responsible for adapting it to
their own conventions.
"""

from __future__ import annotations

import warnings
from enum import auto
from typing import TYPE_CHECKING, Literal

import numpy as np

from napari.utils.misc import StringEnum

if TYPE_CHECKING:
    import numpy.typing as npt


class VerticalAxisOrientation(StringEnum):
    UP = auto()
    DOWN = auto()


class HorizontalAxisOrientation(StringEnum):
    LEFT = auto()
    RIGHT = auto()


class DepthAxisOrientation(StringEnum):
    AWAY = auto()
    TOWARDS = auto()


class Handedness(StringEnum):
    RIGHT = auto()
    LEFT = auto()


VerticalAxisOrientationStr = Literal['up', 'down']
HorizontalAxisOrientationStr = Literal['left', 'right']
DepthAxisOrientationStr = Literal['away', 'towards']

# Prior to v0.6.0, the default would be equivalent to ('away', 'down', 'right')
DEFAULT_ORIENTATION_TYPED = (
    DepthAxisOrientation.TOWARDS,
    VerticalAxisOrientation.DOWN,
    HorizontalAxisOrientation.RIGHT,
)
DEFAULT_ORIENTATION = tuple(map(str, DEFAULT_ORIENTATION_TYPED))


def _base_directions(
    orientation: tuple[
        DepthAxisOrientation,
        VerticalAxisOrientation,
        HorizontalAxisOrientation,
    ],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the camera view and up directions for the home view.

    At the home view (all angles zero) the camera is aligned with the napari
    axes: the view direction lies along the depth axis (dim0) and the up
    direction along the vertical axis (dim1), with signs following the
    orientation. The right direction closes the basis as ``view x up``; it is
    not returned here because the horizontal orientation only affects how the
    renderer mirrors the image, not the view/up math.

    Parameters
    ----------
    orientation : 3-tuple of str
        The napari orientation, with depth, vertical, and horizontal components,
        in napari (zyx) order.

    Returns
    -------
    tuple of np.ndarray
        The view and up directions of the home view.
    """
    depth, vertical, _ = orientation
    view = np.array([-1 if str(depth) == 'towards' else 1, 0, 0], dtype=float)
    up = np.array([0, -1 if str(vertical) == 'down' else 1, 0], dtype=float)
    return view, up


def _camera_rotation_matrix(
    angles: tuple[float, float, float],
) -> npt.NDArray[np.float64]:
    """Return the camera rotation matrix for the given Euler angles.

    The angles are interpreted as Euler rotations about the axes of the three
    displayed dimensions: the camera is rotated about the depth axis (dim0) by
    the first angle, about the vertical axis (dim1) by the second, and about
    the horizontal axis (dim2) by the third. In napari (zyx) coordinates these
    are the z, y, and x axes respectively, so the rotation is
    ``R = Rx(rx) @ Ry(ry) @ Rz(rz)`` (right-handed, i.e. counterclockwise when
    looking along the axis toward the viewer). The rotation maps the home-view
    camera directions (see :func:`_base_directions`) onto the directions for
    the given angles.

    Parameters
    ----------
    angles : 3-tuple of float
        Euler angles of the camera in 3D viewing, in degrees.

    Returns
    -------
    np.ndarray
        The (3, 3) rotation matrix in napari (zyx) coordinates.
    """
    rx, ry, rz = np.radians(angles)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
    rot_y = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    rot_z = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])
    return rot_x @ rot_y @ rot_z


def view_direction_from_angles(
    angles: tuple[float, float, float],
    orientation: tuple[
        DepthAxisOrientation,
        VerticalAxisOrientation,
        HorizontalAxisOrientation,
    ],
) -> tuple[float, float, float]:
    """Return the 3D view direction vector for the given angles.

    The direction is returned in 3D scene coordinates (world coordinates of
    the three displayed dimensions). With all angles zero it matches the
    home-view direction for the given orientation.
    """
    base_view, _ = _base_directions(orientation)
    return tuple(_camera_rotation_matrix(angles) @ base_view)


def up_direction_from_angles(
    angles: tuple[float, float, float],
    orientation: tuple[
        DepthAxisOrientation,
        VerticalAxisOrientation,
        HorizontalAxisOrientation,
    ],
) -> tuple[float, float, float]:
    """Return the 3D up direction vector for the given angles.

    The direction is returned in 3D scene coordinates (world coordinates of
    the three displayed dimensions). With all angles zero it matches the
    home-view direction for the given orientation.
    """
    _, base_up = _base_directions(orientation)
    return tuple(_camera_rotation_matrix(angles) @ base_up)


def angles_from_view_direction(
    view_direction: tuple[float, float, float],
    up_direction: tuple[float, float, float],
    orientation: tuple[
        DepthAxisOrientation,
        VerticalAxisOrientation,
        HorizontalAxisOrientation,
    ],
) -> tuple[float, float, float]:
    """Return camera Euler angles matching the given direction vectors.

    The inverse of :func:`view_direction_from_angles` and
    :func:`up_direction_from_angles` (which therefore produce the same view and
    up directions for the returned angles).

    Parameters
    ----------
    view_direction : 3-tuple of float
        The desired view direction in 3D scene coordinates.
    up_direction : 3-tuple of float
        A direction vector which will point upwards on the canvas. It must not
        be parallel to the ``view_direction`` and does not need to be orthogonal
        to it; it will be projected.
    orientation : 3-tuple of str
        The napari orientation, with depth, vertical, and horizontal components,
        in napari (zyx) order.

    Returns
    -------
    3-tuple of float
        Euler angles (rx, ry, rz) of the camera in 3D viewing, in degrees.
    """
    from scipy.spatial.transform import Rotation as R

    view = np.asarray(view_direction, dtype=float)
    view = view / np.linalg.norm(view)
    up = np.asarray(up_direction, dtype=float)
    up = up - np.dot(up, view) * view
    up = up / np.linalg.norm(up)

    # The rotation maps the home-view camera basis onto the given view/up
    # basis; the right direction closes each basis. The columns of a basis are
    # the view, up, and right directions, so the rotation is recovered as
    # ``camera_basis @ home_basis.T`` (both bases are orthonormal).
    base_view, base_up = _base_directions(orientation)
    camera_basis = np.stack([view, up, np.cross(view, up)], axis=1)
    home_basis = np.stack(
        [base_view, base_up, np.cross(base_view, base_up)], axis=1
    )
    rotation = camera_basis @ home_basis.T

    # The rotation is ``Rx(rx) @ Ry(ry) @ Rz(rz)`` (see
    # :func:`_camera_rotation_matrix`), which scipy's extrinsic 'XYZ' sequence
    # reports as the negated (-rx, -ry, -rz) triple, hence the negation.
    # Gimbal lock is an expected, handled edge case here (scipy sets the third
    # angle to zero), so the associated warning is not actionable.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        angles = R.from_matrix(rotation).as_euler('XYZ', degrees=True)
    return tuple(-angles)
