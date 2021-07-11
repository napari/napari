from typing import Tuple

from pydantic import validator

from napari.utils.events import EventedModel
from napari.utils.misc import ensure_n_tuple


class Plane3D(EventedModel):
    """Object modelling a plane in 3D with a defined thickness.

    Attributes
    ----------
    position : 3-tuple
        A position in 3D, defined in data coordinates.
    normal_vector : 3-tuple
        A normal vector in 3D, defined in data coordinates.
    thickness : float
        A thickness for the slice
    """

    position: Tuple[float, float, float] = (0, 0, 0)
    normal_vector: Tuple[float, float, float] = (1, 0, 0)
    thickness: float = 10.0

    @validator('position', 'normal_vector', pre=True)
    def _ensure_3_tuple(cls, v):
        return ensure_n_tuple(v, n=3)
