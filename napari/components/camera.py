from dataclasses import InitVar
from typing import ClassVar, Tuple

from ..utils.events.dataclass import Property, dataclass


@dataclass(events=True, properties=True)
class Camera:
    """Camera object modeling position and view of the camera.

    Parameters
    ----------
    dims : napari.components.Dims
        Dims model of the viewer.
    angles : 3-tuple
        Euler angles of camera in 3D viewing (rx, ry, rz), in degrees.
        Only used during 3D viewing.
    center : 2-tuple or 3-tuple
        Center of the camera for either 2D or 3D viewing.
    zoom : float
        Scale from canvas pixels to world pixels.

    Attributes
    ----------
    angles : 3-tuple
        Euler angles of camera in 3D viewing (rx, ry, rz), in degrees.
        Only used during 3D viewing.
    center : 2-tuple or 3-tuple
        Center of the camera for either 2D or 3D viewing.
    ndisplay : int
        Number of displayed dimensions, must be either 2 or 3.
    zoom : float
        Scale from canvas pixels to world pixels.
    """

    dims: InitVar
    _dims: ClassVar

    center: Property[Tuple, None, tuple] = (0, 0, 0)
    zoom: int = 1
    angles: int = (0, 0, 90)

    def __post_init__(self, dims):
        self._dims = dims
        self.center = self.center[-self._dims.ndisplay :]

    # Should we include this error checking when setting the center?
    # @center.setter
    # def center(self, center):
    #     if self.center == tuple(center):
    #         return
    #     if self.ndisplay != len(center):
    #         raise ValueError(
    #             f'Center must be same length as currently displayed'
    #             f' dimensions, got {len(center)} need {self.ndisplay}.'
    #         )
    #     self._center = tuple(center)
    #     self.events.center()

    @property
    def ndisplay(self):
        """int: Dimensionality of the camera rendering."""
        return self._dims.ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay):
        self._dims.ndisplay = ndisplay

    # Should we include this special casing when ndisplay=2?
    # @property
    # def angles(self):
    #     """3-tuple: Euler angles of camera in 3D viewing, in degrees."""
    #     if self.ndisplay == 3:
    #         return self._angles
    #     else:
    #         return (0, 0, 90)
