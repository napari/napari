from functools import partial
from typing import Tuple

from ..utils.events.dataclass import Property, evented_dataclass
from ..utils.misc import ensure_n_tuple

Len_3_Tuple = Tuple[float, float, float]
ensure_3_tuple = partial(ensure_n_tuple, n=3)


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
    interactive : bool
        If the camera interactivity is enabled or not.
    """

    center: Property[Len_3_Tuple, None, ensure_3_tuple] = (0, 0, 0)
    zoom: float = 1
    angles: Property[Len_3_Tuple, None, ensure_3_tuple] = (0, 0, 90)
    interactive: bool = True
