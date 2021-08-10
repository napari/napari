from typing import Tuple

import numpy as np
from pydantic import validator

from ...utils.events import EventedModel
from ...utils.geometry import intersect_line_with_plane_3d


class PlaneManager(EventedModel):
    """Manages properties relating to plane rendering in 3D.

    In this object, planes in 3D are defined by a position, a normal vector
    and a thickness parameter.

    Attributes
    ----------
    position : 3-tuple
        A 3D position on the plane, defined in data coordinates.
    normal : 3-tuple
        A 3D unit vector normal to the plane, defined in data coordinates.
    thickness : float
        Thickness of the slice
    enabled : bool
        Whether plane rendering is enabled.
    """

    position: Tuple[float, float, float] = (0, 0, 0)
    normal: Tuple[float, float, float] = (1, 0, 0)
    thickness: float = 10.0
    enabled: bool = False

    @validator('normal')
    def _normalise_vector(cls, v):
        return tuple(v / np.linalg.norm(v))

    @validator('normal', 'position', pre=True)
    def _ensure_tuple(cls, v):
        return tuple(v)

    def shift_along_normal_vector(self, distance: float):
        """Shift the plane along its normal vector by a given distance."""
        self.position += distance * self.normal

    def intersect_with_line(
        self, line_position: np.ndarray, line_direction: np.ndarray
    ) -> np.ndarray:
        """Calculate a 3D line-plane intersection."""
        return intersect_line_with_plane_3d(
            line_position, line_direction, self.position, self.normal
        )

    @classmethod
    def from_points(cls, a, b, c):
        """Derive a PlaneManager from three points.

        Parameters
        ----------
        a : ArrayLike
            (3,) array containing coordinates of a point
        b : ArrayLike
            (3,) array containing coordinates of a point
        c : ArrayLike
            (3,) array containing coordinates of a point

        Returns
        -------
        plane : PlaneManager
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        abc = np.row_stack((a, b, c))
        ab = b - a
        ac = c - a

        plane_normal = np.cross(ab, ac)
        plane_position = np.mean(abc, axis=0)
        return cls(position=plane_position, normal=plane_normal)
