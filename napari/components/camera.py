from typing import Tuple

from ..utils.events.dataclass import Property, dataclass


@dataclass(events=True, properties=True)
class Camera:
    """Camera object modeling position and view of the camera.

    Parameters
    ----------
    angles : 3-tuple
        Euler angles of camera in 3D viewing (rx, ry, rz), in degrees.
        Only used during 3D viewing.
    center : 2-tuple or 3-tuple
        Center of the camera for either 2D or 3D viewing.
    zoom : float
        Scale from canvas pixels to world pixels.
    """

    center: Property[Tuple, None, tuple] = (0, 0, 0)
    zoom: int = 1
    angles: Property[Tuple, None, tuple] = (0, 0, 90)
