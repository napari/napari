from typing import Tuple

from ..utils.events.dataclass import Property, evented_dataclass
from ..utils.misc import force_3_tuple

_3_Tuple = Tuple[float, float, float]


@evented_dataclass
class Camera:
    """Camera object modeling position and view of the camera.

    Attributes
    ----------
    center : 3-tuple
        Center of the camera. In 2D viewing the last two values are used.
    zoom : float
        Scale from canvas pixels to world pixels.
    angles : 3-tuple
        Euler angles of camera in 3D viewing (rx, ry, rz), in degrees.
        Only used during 3D viewing.
    """

    center: Property[_3_Tuple, None, force_3_tuple] = (0,) * 3
    zoom: float = 1
    angles: Property[_3_Tuple, None, force_3_tuple] = (0,) * 3
