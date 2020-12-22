from functools import partial
from typing import Tuple

from pydantic import BaseModel, validator

from ..utils.events.event_utils import evented
from ..utils.misc import ensure_n_tuple

Float_3_Tuple = Tuple[float, float, float]
ensure_3_tuple = partial(ensure_n_tuple, n=3)


@evented
class Camera(BaseModel):
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

    center: Float_3_Tuple = (0.0, 0.0, 0.0)
    zoom: float = 1.0
    angles: Float_3_Tuple = (0.0, 0.0, 90.0)
    interactive: bool = True

    class Config:
        validate_assignment = True

    @validator('center', 'angles', pre=True)
    def _ensure_3_tuple(cls, v):
        return ensure_3_tuple(v)
