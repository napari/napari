from typing import Tuple

from pydantic import validator

from napari.utils.events import EventedModel
from napari.utils.misc import ensure_n_tuple


class Plane(EventedModel):
    """Plane object modelling a plane in 3D.

    A plane is defined by a position and a normal vector.

    Attributes
    ----------
    position : 3-tuple
        A position on the plane
    normal : 3-tuple
        A normal vector which defines the orientation of the plane

    """

    position: Tuple[float, float, float]
    normal: Tuple[float, float, float]

    @validator('position', 'normal', pre=True)
    def _ensure_3_tuple(cls, v):
        return ensure_n_tuple(v, n=3)


class Slice(EventedModel):
    """Slice object modelling a plane with a defined thickness in 3D.

    Attributes
    ----------

    plane : Plane
        A plane in 3D.
    thickness : float
        A thickness for the slice
    """

    plane: Plane
    thickness: float
