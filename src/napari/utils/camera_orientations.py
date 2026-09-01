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


def _base_directions(
    orientation: AxesOrientation3D,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Home-view camera view and up directions for the given orientation.

    The horizontal orientation only affects how the renderer mirrors the
    image, so only the view and up directions are returned.
    """
    depth, vertical, _ = orientation
    view = np.array([-1 if str(depth) == 'towards' else 1, 0, 0], dtype=float)
    up = np.array([0, -1 if str(vertical) == 'down' else 1, 0], dtype=float)
    return view, up


def _camera_rotation_matrix(
    angles: tuple[float, float, float],
) -> npt.NDArray[np.float64]:
    """Return the camera rotation matrix for the given Euler angles.

    The camera is rotated about the depth (dim0), vertical (dim1), and
    horizontal (dim2) axes by the first, second, and third angles, in that
    order.
    """
    from scipy.spatial.transform import Rotation as R

    # scipy's 'XYZ' sequence rotates opposite to the napari convention.
    return R.from_euler('XYZ', -np.asarray(angles), degrees=True).as_matrix()


def view_direction_from_angles(
    angles: tuple[float, float, float],
    orientation: AxesOrientation3D,
) -> tuple[float, float, float]:
    """Return the 3D view direction for the given angles.

    The direction is in 3D scene coordinates (world coordinates of the three
    displayed dimensions).
    """
    base_view, _ = _base_directions(orientation)
    return tuple(_camera_rotation_matrix(angles) @ base_view)


def up_direction_from_angles(
    angles: tuple[float, float, float],
    orientation: AxesOrientation3D,
) -> tuple[float, float, float]:
    """Return the 3D up direction for the given angles.

    The direction is in 3D scene coordinates (world coordinates of the three
    displayed dimensions).
    """
    _, base_up = _base_directions(orientation)
    return tuple(_camera_rotation_matrix(angles) @ base_up)


def angles_from_view_direction(
    view_direction: tuple[float, float, float],
    up_direction: tuple[float, float, float],
    orientation: AxesOrientation3D,
) -> tuple[float, float, float]:
    """Return camera Euler angles matching the given direction vectors.

    The inverse of :func:`view_direction_from_angles` and
    :func:`up_direction_from_angles`.

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

    # The rotation maps the home-view basis onto the given view/up basis:
    # ``rotation = camera_basis @ home_basis.T`` (bases are orthonormal).
    base_view, base_up = _base_directions(orientation)
    camera_basis = np.stack([view, up, np.cross(view, up)], axis=1)
    home_basis = np.stack(
        [base_view, base_up, np.cross(base_view, base_up)], axis=1
    )
    rotation = camera_basis @ home_basis.T

    # scipy 'XYZ' reports the angles negated; gimbal-lock warnings here are
    # expected and harmless.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        angles = R.from_matrix(rotation).as_euler('XYZ', degrees=True)
    return tuple(-angles)
