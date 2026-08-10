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


def _orientation_signs(
    orientation: tuple[
        DepthAxisOrientation,
        VerticalAxisOrientation,
        HorizontalAxisOrientation,
    ],
) -> npt.NDArray[np.float64]:
    """Sign factors (-1 or 1) for each camera angle based on current orientation."""
    return np.array(
        [
            1 if ori == default_ori else -1
            for ori, default_ori in zip(
                orientation, DEFAULT_ORIENTATION, strict=True
            )
        ]
    )


def _camera_rotation_matrix(
    angles: tuple[float, float, float],
    orientation: tuple[
        DepthAxisOrientation,
        VerticalAxisOrientation,
        HorizontalAxisOrientation,
    ],
) -> npt.NDArray[np.float64]:
    """Return the 3D camera basis vectors as rows of a rotation matrix.

    The returned matrix ``M`` is a rotation matrix in napari (zyx) coordinates
    whose rows, after the orientation-dependent sign flips, correspond to the
    camera up, view, and right directions (rows 0, 1, and 2, respectively).

    The camera rotation is an extrinsic Euler rotation applied in ``yxz`` order
    about the three displayed dimensions, combined with a fixed 90-degree
    rotation for the home view. See #8281 for the historical origin of this
    convention.

    Parameters
    ----------
    angles : 3-tuple of float
        Euler angles of the camera in 3D viewing, in degrees.
    orientation : 3-tuple of str
        The napari orientation, with depth, vertical, and horizontal components,
        in napari (zyx) order.

    Returns
    -------
    np.ndarray
        The (3, 3) rotation matrix with rows in napari (zyx) order.
    """
    from scipy.spatial.transform import Rotation as R

    angle_factors = _orientation_signs(orientation)
    rotation = R.from_euler(
        'yxz', np.asarray(angles) * angle_factors, degrees=True
    )
    # The home view (all angles zero) is a 90-degree rotation about the z axis.
    home_rotation = R.from_euler('z', 90, degrees=True).as_matrix()
    return rotation.as_matrix() @ home_rotation


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
    the three displayed dimensions).
    """
    # directions are based on the *opposite* angle, so we invert
    direction_factors = _orientation_signs(orientation)[::-1]
    matrix = _camera_rotation_matrix(angles, orientation)
    return tuple(direction_factors * matrix[1])


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
    the three displayed dimensions).
    """
    # directions are based on the *opposite* angle, so we invert
    direction_factors = _orientation_signs(orientation)
    matrix = _camera_rotation_matrix(angles, orientation)
    return tuple(direction_factors * matrix[0])


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

    # The rows of the camera rotation matrix (in napari coordinates) are the
    # up, view, and right directions, scaled elementwise by the direction sign
    # factors (see the forward direction in `_camera_rotation_matrix`). The
    # right direction closes the basis.
    angle_factors = _orientation_signs(orientation)
    direction_factors = angle_factors[::-1]
    up_row = direction_factors * up
    view_row = direction_factors * view
    matrix = np.stack([up_row, view_row, np.cross(up_row, view_row)])

    # Undo the home-view rotation, then invert the ``yxz`` Euler rotation.
    home_rotation = R.from_euler('z', 90, degrees=True).as_matrix()
    rotation = matrix @ home_rotation.T
    # Gimbal lock is an expected, handled edge case here (scipy sets the third
    # angle to zero), so the associated warning is not actionable.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        angles = R.from_matrix(rotation).as_euler('yxz', degrees=True)
    return tuple(angles * angle_factors)
