from typing import Tuple

from pydantic import validator

from ..utils.events import EventedModel
from ..utils.misc import ensure_n_tuple


class Camera(EventedModel):
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

    # fields
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    zoom: float = 1.0
    angles: Tuple[float, float, float] = (0.0, 0.0, 90.0)
    interactive: bool = True

    # validators
    @validator('center', 'angles', pre=True)
    def _ensure_3_tuple(v):
        return ensure_n_tuple(v, n=3)
