from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from vispy.util.quaternion import Quaternion


def quaternion2euler_degrees(
    quaternion: Quaternion,
) -> tuple[float, float, float]:
    """Converts VisPy quaternion into euler angle representation.

    Euler angles have degeneracies, so the output might different
    from the Euler angles that might have been used to generate
    the input quaternion.

    Euler angles representation also has a singularity
    near pitch = Pi/2 ; to avoid this, we set to Pi/2 pitch angles
    that are closer than the chosen epsilon from it.

    Parameters
    ----------
    quaternion : vispy.util.Quaternion
        Quaternion for conversion.

    Returns
    -------
    angles : 3-tuple
        Euler angles in (rx, ry, rz) order, in degrees.
    """
    epsilon = 1e-10

    q = quaternion

    sin_theta_2 = 2 * (q.w * q.y - q.z * q.x)
    sin_theta_2 = np.sign(sin_theta_2) * min(abs(sin_theta_2), 1)

    if abs(sin_theta_2) > 1 - epsilon:
        theta_1 = -np.sign(sin_theta_2) * 2 * np.arctan2(q.x, q.w)
        theta_2 = np.arcsin(sin_theta_2)
        theta_3 = 0

    else:
        theta_1 = np.arctan2(
            2 * (q.w * q.z + q.y * q.x),
            1 - 2 * (q.y * q.y + q.z * q.z),
        )

        theta_2 = np.arcsin(sin_theta_2)

        theta_3 = np.arctan2(
            2 * (q.w * q.x + q.y * q.z),
            1 - 2 * (q.x * q.x + q.y * q.y),
        )

    angles = (theta_1, theta_2, theta_3)

    return cast(tuple[float, float, float], tuple(np.degrees(angles)))
