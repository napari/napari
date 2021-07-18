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

    @property
    def normalised_normal_vector(self):
        return self.normal_vector / np.linalg.norm(self.normal_vector)

    def shift_along_normal_vector(self, distance: float):
        """Shift the plane along its normal vector by a given distance."""
        self.position += distance * self.normalised_normal_vector
