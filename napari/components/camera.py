from typing import Tuple

import numpy as np
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
        Note that Euler angles's intrinsic degeneracy means different
        sets of Euler angles may lead to the same view.
    perspective : float
        Perspective (aka "field of view" in vispy) of the camera (if 3D).
    interactive : bool
        If the camera interactivity is enabled or not.
    """

    # fields
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    zoom: float = 1.0
    angles: Tuple[float, float, float] = (0.0, 0.0, 90.0)
    perspective: float = 0
    interactive: bool = True

    # validators
    @validator('center', 'angles', pre=True)
    def _ensure_3_tuple(v):
        return ensure_n_tuple(v, n=3)

    @property
    def view_direction(self) -> np.ndarray:
        """3D view direction vector of the camera.

        View direction is calculated from the Euler angles and returned as a
        (3,) array.
        """
        ang = np.deg2rad(self.angles)
        view_direction = (
            np.sin(ang[2]) * np.cos(ang[1]),
            np.cos(ang[2]) * np.cos(ang[1]),
            -np.sin(ang[1]),
        )
        return view_direction

    def calculate_nd_view_direction(
        self, ndim: int, dims_displayed: Tuple[int]
    ) -> np.ndarray:
        """Calculate the nD view direction vector of the camera.

        Parameters
        ----------
        ndim : int
            Number of dimensions in which to embed the 3D view vector.
        dims_displayed : Tuple[int]
            Dimensions in which to embed the 3D view vector.


        Returns
        -------
        view_direction_nd : np.ndarray
            nD view direction vector as an (ndim, ) ndarray

        """
        if len(dims_displayed) != 3:
            return None
        view_direction_nd = np.zeros(ndim)
        view_direction_nd[list(dims_displayed)] = self.view_direction
        return view_direction_nd
