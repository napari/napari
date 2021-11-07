import inspect
from typing import Tuple, no_type_check

import numpy as np
from pydantic import validator
from scipy.spatial.transform import Rotation as R

from ..utils.events import EventedModel
from ..utils.misc import ensure_n_tuple


class Camera(EventedModel):
    """Camera object modeling position and view of the camera.

    Attributes
    ----------
    center : 3-tuple
        Center of rotation for the camera.
        In 2D viewing the last two values are used.
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

    @view_direction.setter
    def view_direction(self, view_direction: Tuple[float, float, float]):
        if (view_direction[0], view_direction[2]) == (0, 0):
            up_direction = (-1, 0, 0)
        else:
            up_direction = (0, -1, 0)
        self.set_view_direction(
            view_direction=view_direction, up_direction=up_direction
        )

    def set_view_direction(
        self,
        view_direction: Tuple[float, float, float],
        up_direction: Tuple[float, float, float] = (0, -1, 0),
    ):
        # xyz ordering for vispy, normalise vectors for rotation matrix
        view_direction = np.asarray(view_direction, dtype=float)[::-1]
        view_direction /= np.linalg.norm(view_direction)

        up_direction = np.asarray(up_direction, dtype=float)[::-1]
        up_direction = np.cross(view_direction, up_direction)
        up_direction /= np.linalg.norm(up_direction)

        # explicit check for parallel view direction and up direction
        if np.allclose(np.cross(view_direction, up_direction), 0):
            raise ValueError("view direction and up direction are parallel")

        x_direction = np.cross(up_direction, view_direction)
        x_direction /= np.linalg.norm(x_direction)

        # construct rotation matrix, convert to euler angles
        rotation_matrix = np.column_stack(
            (up_direction, view_direction, x_direction)
        )
        euler_angles = R.from_matrix(rotation_matrix).as_euler(
            seq='yzx', degrees=True
        )
        self.angles = euler_angles

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

    @no_type_check
    def __setattr__(self, name, value):
        """Enable use of properties with setters in pydantic models."""
        try:
            super().__setattr__(name, value)
        except ValueError as e:
            setters = inspect.getmembers(
                self.__class__,
                predicate=lambda x: isinstance(x, property)
                and x.fset is not None,
            )
            for setter_name, func in setters:
                if setter_name == name:
                    object.__setattr__(self, name, value)
                    break
            else:
                raise e
