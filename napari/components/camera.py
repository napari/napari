from ..utils.events import EmitterGroup


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

    def __init__(self, dims, *, center=None, zoom=1, angles=(0, 0, 90)):
        self._dims = dims

        if center is None:
            center = (0,) * self._dims.ndisplay
        center = center[-self._dims.ndisplay :]

        self._zoom = zoom
        self._angles = angles
        self._center = center

        self.events = EmitterGroup(
            source=self,
            auto_connect=True,
            angles=None,
            zoom=None,
            center=None,
        )

    @property
    def center(self):
        """tuple: Center point of camera view for 2D or 3D viewing."""
        return self._center

    @center.setter
    def center(self, center):
        if self.center == tuple(center):
            return
        if self.ndisplay != len(center):
            raise ValueError(
                f'Center must be same length as currently displayed'
                f' dimensions, got {len(center)} need {self.ndisplay}.'
            )
        self._center = tuple(center)
        self.events.center()

    @property
    def ndisplay(self):
        """int: Dimensionality of the camera rendering."""
        return self._dims.ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay):
        self._dims.ndisplay = ndisplay

    @property
    def zoom(self):
        """float: Scale from canvas pixels to world pixels."""
        return self._zoom

    @zoom.setter
    def zoom(self, zoom):
        if self._zoom == zoom:
            return
        self._zoom = zoom
        self.events.zoom()

    @property
    def angles(self):
        """3-tuple: Euler angles of camera in 3D viewing, in degrees."""
        if self.ndisplay == 3:
            return self._angles
        else:
            return (0, 0, 90)

    @angles.setter
    def angles(self, angles):
        if self._angles == tuple(angles):
            return
        self._angles = tuple(angles)
        self.events.angles()
