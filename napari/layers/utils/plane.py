from typing import Tuple

import numpy as np
from pydantic import validator

from ...utils.events import EventedModel
from ...utils.geometry import intersect_line_with_plane_3d
from ...utils.misc import ensure_n_tuple


class PlaneManager(EventedModel):
    """Manages properties relating to a plane in 3D with a defined thickness.

    Attributes
    ----------
    position : 3-tuple
        A 3D position on the plane, defined in data coordinates.
    normal_vector : 3-tuple
        A 3D unit vector normal to the plane, defined in data coordinates.
    thickness : float
        Thickness of the slice
    enabled : bool
        Whether the plane is enabled.
    """

    position: Tuple[float, float, float] = (0, 0, 0)
    normal_vector: Tuple[float, float, float] = (1, 0, 0)
    thickness: float = 10.0
    enabled: bool = False

    @validator('position', 'normal_vector', pre=True)
    def _ensure_3_tuple(cls, v):
        return ensure_n_tuple(v, n=3)

    @validator('normal_vector')
    def _ensure_normalised_vector(cls, v):
        return tuple(v / np.linalg.norm(v))

    def shift_along_normal_vector(self, distance: float):
        """Shift the plane along its normal vector by a given distance."""
        self.position += distance * self.normal_vector

    def intersect_with_line(
        self, line_position: np.ndarray, line_orientation: np.ndarray
    ) -> np.ndarray:
        """Calculate a 3D line-plane intersection."""
        return intersect_line_with_plane_3d(
            line_position, line_orientation, self.position, self.normal_vector
        )
