from __future__ import annotations

import warnings
from enum import auto
from typing import TYPE_CHECKING, Literal, TypeAlias

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


AxisOrientation: TypeAlias = type[
    DepthAxisOrientation | VerticalAxisOrientation | HorizontalAxisOrientation
]


AxesOrientation3D: TypeAlias = tuple[
    DepthAxisOrientation,
    VerticalAxisOrientation,
    HorizontalAxisOrientation,
]

AxesOrientation2D: TypeAlias = tuple[
    VerticalAxisOrientation,
    HorizontalAxisOrientation,
]

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

Vector3D: TypeAlias = tuple[float, float, float]
Vector2D: TypeAlias = tuple[float, float]


def _base_view_and_up_direction(
    orientation: AxesOrientation3D,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """View and Up directions for the base view, given a certain Orientation setting."""
    depth, vertical, _ = orientation
    view = np.array([-1 if str(depth) == 'towards' else 1, 0, 0], dtype=float)
    up = np.array([0, -1 if str(vertical) == 'down' else 1, 0], dtype=float)
    return view, up


def _camera_rotation_matrix(
    angles: Vector3D,
) -> npt.NDArray[np.float64]:
    """Return the camera rotation matrix for the given Euler angles."""
    from scipy.spatial.transform import Rotation as R

    # we take the opposite of the angles to match scipy's XYZ conventions
    # while retaining napari behaviour
    return R.from_euler('XYZ', -np.asarray(angles), degrees=True).as_matrix()


def view_and_up_directions_from_angles(
    angles: Vector3D,
    orientation: AxesOrientation3D,
) -> tuple[Vector3D, Vector3D]:
    """Return the 3D view and up direction for the given angles.

    The directions are in 3D scene coordinates (world coordinates of the three
    displayed dimensions).

    Parameters
    ----------
    angles: 3-tuple of float
        Euler angles (rx, ry, rz) of the camera in 3D viewing, in degrees.
    orientation : 3-tuple of str
        The orientation, with depth, vertical, and horizontal components,
        in napari (zyx) order.

    Returns
    -------
    view_direction : 3-tuple of float
        The view direction in 3D scene coordinates.
    up_direction : 3-tuple of float
        The up direction in 3D scene coordinates.
    """
    base_view, base_up = _base_view_and_up_direction(orientation)
    rot_matrix = _camera_rotation_matrix(angles)
    return tuple(rot_matrix @ base_view), tuple(rot_matrix @ base_up)


def angles_from_view_and_up_directions(
    view_direction: Vector3D,
    up_direction: Vector3D,
    orientation: AxesOrientation3D,
) -> Vector3D:
    """Return camera Euler angles matching the given direction vectors.

    The inverse of :func:`view_and_up_directions_from_angles`.

    Parameters
    ----------
    view_direction : 3-tuple of float
        The desired view direction in 3D scene coordinates.
    up_direction : 3-tuple of float
        A direction vector which will point upwards on the canvas. It must not
        be parallel to the ``view_direction`` and does not need to be orthogonal
        to it; it will be projected.
    orientation : 3-tuple of str
        The orientation, with depth, vertical, and horizontal components,
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

    # the rotation maps the home-view basis onto the given view/up basis
    base_view, base_up = _base_view_and_up_direction(orientation)
    camera_basis = np.stack([view, up, np.cross(view, up)], axis=1)
    home_basis = np.stack(
        [base_view, base_up, np.cross(base_view, base_up)], axis=1
    )
    rotation = camera_basis @ home_basis.T

    # gimbal locks are expected, not an issue
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        angles = R.from_matrix(rotation).as_euler('XYZ', degrees=True)
    # scipy has inverted convention
    return tuple(-angles)
