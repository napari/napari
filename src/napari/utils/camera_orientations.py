from __future__ import annotations

import warnings
from enum import auto
from typing import TYPE_CHECKING, Literal, cast

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


def _get_vispy_flipped_axes(
    orientation: tuple[
        DepthAxisOrientation,
        VerticalAxisOrientation,
        HorizontalAxisOrientation,
    ],
    ndisplay: Literal[2, 3] = 2,
) -> tuple[int, int, int]:
    """Return the VisPy axis flips corresponding to the given orientation.

    Parameters
    ----------
    orientation : 3-tuple of str
        The napari orientation, with depth, vertical, and horizontal components,
        in napari (zyx) order.
    ndisplay : {2, 3}
        Whether the flips are for the 2D or 3D VisPy camera.

    Returns
    -------
    3-tuple of int
        The VisPy flips, in VisPy (xyz) order.
    """
    # Note: the Vispy axis order is xyz, or horizontal, vertical, depth,
    # while the napari axis order is zyx / plane-row-column, or depth, vertical,
    # horizontal — i.e. it is exactly inverted. This switch happens when data
    # is passed from napari to Vispy, usually with a transposition. In the camera
    # models, this means that the order of these orientations appear in the
    # opposite order to that in napari.components.Camera.
    #
    # Note that the default Vispy camera orientations come from Vispy, not from us.
    vispy_default_orientation = (
        ('right', 'up', 'towards')
        if ndisplay == 2
        else ('right', 'down', 'away')
    )

    # Vispy uses xyz coordinates; napari uses zyx coordinates. We therefore
    # start by inverting the order of coordinates coming from the napari
    # camera model:
    orientation_xyz = orientation[::-1]
    # The Vispy camera flip is a tuple of three ints in {0, 1}, indicating
    # whether they are flipped relative to the Vispy default.
    return cast(
        tuple[int, int, int],
        tuple(
            int(ori != default_ori)
            for ori, default_ori in zip(
                orientation_xyz, vispy_default_orientation, strict=True
            )
        ),
    )


def _flipped_axes_to_factors(
    flipped_axes: tuple[int, int, int],
) -> npt.NDArray[np.float64]:
    """Convert VisPy axis flips to an array of +/-1 factors.

    A flipped axis corresponds to a factor of -1, an unflipped axis to +1.
    """
    return np.where(flipped_axes, -1, 1)


def _angles_to_vispy_rotation_matrix(
    angles: tuple[float, float, float],
    flipped_axes: tuple[int, int, int],
) -> npt.NDArray[np.float64]:
    """Replicate the VisPy camera rotation matrix from napari angles.

    This reproduces the transformation applied to the 3D camera by VisPy for
    the given napari Euler ``angles`` and axis ``flipped_axes``, without
    requiring a VisPy scene. See ``napari_angles_to_vispy_quat`` in
    ``napari._vispy.camera`` for the underlying logic.

    Returns
    -------
    np.ndarray
        The (3, 3) rotation matrix with rows in VisPy (xyz) order, where the
        rows correspond to the camera right, view, and up directions.
    """
    from scipy.spatial.transform import Rotation as R

    # flip handedness so the rotation is always righthanded even with axis flipping
    angles_flipped = angles * np.where(flipped_axes, -1, 1)
    # undo vispy quirks (rotation of 90 digrees and lefthanded y axis)
    angles_fixed = (np.array(angles_flipped) * (1, -1, 1)) + (0, 0, 90)
    # see #8281 for why this is yzx. In short: longstanding vispy bug.
    w, x, y, z = R.from_euler('yzx', angles_fixed, degrees=True).as_quat(
        scalar_first=True
    )
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    norm = (x * x + y * y + z * z) ** 0.5
    ax, ay, az = (x / norm, y / norm, z / norm) if norm else (1.0, 0.0, 0.0)
    # VisPy rotates about the axis (x, z, y) and its `rotate` method returns
    # the transpose of the standard rotation matrix.
    return R.from_rotvec(angle * np.array([ax, az, ay])).as_matrix().T


def _direction_from_angles(
    angles: tuple[float, float, float],
    flipped_axes: tuple[int, int, int],
    row: int,
) -> tuple[float, float, float]:
    """Return a camera direction vector for the given angles.

    Parameters
    ----------
    angles : 3-tuple of float
        Euler angles of the camera in 3D viewing, in degrees.
    flipped_axes : 3-tuple of int
        VisPy axis flips, in VisPy (xyz) order.
    row : int
        Which row of the VisPy rotation matrix to use, 0 (right), 1 (view),
        or 2 (up).

    Returns
    -------
    3-tuple of float
        The direction vector in napari (zyx) coordinates.
    """
    factors = _flipped_axes_to_factors(flipped_axes)
    rotation = _angles_to_vispy_rotation_matrix(angles, flipped_axes)
    return tuple((factors * rotation[row, :])[::-1])


def view_direction_from_angles(
    angles: tuple[float, float, float],
    flipped_axes: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Return the 3D view direction vector for the given angles.

    The direction is returned in 3D scene coordinates (world coordinates of
    the three displayed dimensions).
    """
    return _direction_from_angles(angles, flipped_axes, 1)


def up_direction_from_angles(
    angles: tuple[float, float, float],
    flipped_axes: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Return the 3D up direction vector for the given angles.

    The direction is returned in 3D scene coordinates (world coordinates of
    the three displayed dimensions).
    """
    return _direction_from_angles(angles, flipped_axes, 2)


def angles_from_view_direction(
    view_direction: tuple[float, float, float],
    up_direction: tuple[float, float, float],
    flipped_axes: tuple[int, int, int],
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
    flipped_axes : 3-tuple of int
        VisPy axis flips, in VisPy (xyz) order.

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

    # The right direction closes the basis: it is the cross product of the
    # view and up directions, with a sign depending on the flip parity so that
    # the basis has the correct handedness for the given flips.
    right = (-1) ** (sum(flipped_axes) + 1) * np.cross(view, up)

    # The rows of the VisPy rotation matrix (in xyz order) are the three
    # directions, scaled elementwise by the flip factors (see the forward
    # direction in `_direction_from_angles`).
    matrix_xyz = np.stack([right[::-1], view[::-1], up[::-1]])
    factors = _flipped_axes_to_factors(flipped_axes)
    rot_matrix = matrix_xyz * factors[None, :]

    # Invert the VisPy rotation construction: the rotation matrix is built as
    # ``from_rotvec(angle * (x, z, y)).T``, so its transpose's rotvec directly
    # gives the axis and angle used by VisPy.
    rho = R.from_matrix(rot_matrix.T).as_rotvec()
    theta = np.linalg.norm(rho)
    if theta < 1e-12:
        ax, ay, az = 1.0, 0.0, 0.0
    else:
        ax, az, ay = rho / theta
    quat = (
        np.cos(theta / 2),
        ax * np.sin(theta / 2),
        ay * np.sin(theta / 2),
        az * np.sin(theta / 2),
    )
    # Gimbal lock is an expected, handled edge case here (scipy sets the third
    # angle to zero), so the associated warning is not actionable.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        angles_fixed = R.from_quat(quat, scalar_first=True).as_euler(
            'yzx', degrees=True
        )
    angles_flipped = (np.array(angles_fixed) - (0, 0, 90)) * (1, -1, 1)
    return tuple(angles_flipped * np.where(flipped_axes, -1, 1))
