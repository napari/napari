from typing import Tuple

import numpy as np
from pydantic import validator

from ...utils.events import EventedModel
from ...utils.misc import ensure_n_tuple


class Plane3D(EventedModel):
    """Object modelling a plane in 3D with a defined thickness.

    Attributes
    ----------
    position : 3-tuple
        A 3D position on the plane, defined in data coordinates.
    normal_vector : 3-tuple
        A 3D unit vector normal to the plane, defined in data coordinates.
    thickness : float
        Thickness of the slice
    """

    position: Tuple[float, float, float] = (0, 0, 0)
    normal_vector: Tuple[float, float, float] = (1, 0, 0)
    thickness: float = 10.0

    @validator('position', 'normal_vector', pre=True)
    def _ensure_3_tuple(cls, v):
        return ensure_n_tuple(v, n=3)

    @validator('normal_vector')
    def _ensure_normalised_vector(cls, v):
        return tuple(v / np.linalg.norm(v))

    def shift_along_normal_vector(self, distance: float):
        """Shift the plane along its normal vector by a given distance."""
        self.position += distance * self.normal_vector
