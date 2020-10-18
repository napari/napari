from ..utils.events import EmitterGroup


class Camera:
    """Camera object modeling position and view of the camera.

    Parameters
    ----------
    dims : napari.components.Dims
        Dims model of the viewer.
    zoom : float
        Scale from canvas pixels to world pixels.
    angles : 3-tuple
        Euler angles of camera in 3D viewing (rx, ry, rz), in degrees.
        Only used during 3D viewing.

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

    def __init__(self, dims, *, zoom=1, angles=(0, 0, 90)):

        self._dims = dims
        self._zoom = zoom
        self._angles = angles

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
        return tuple(self._dims.point[d] for d in self._dims.displayed)

    @center.setter
    def center(self, center):
        if self.center == tuple(center):
            return
        if self.ndisplay != len(center):
            raise ValueError(
                f'Center must be same length as currently displayed'
                f' dimensions, got {len(center)} need {self.ndisplay}.'
            )
        axes = self._dims.displayed
        for axis, value in zip(axes, center):
            self._dims.set_point(axis, value)
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
        return self._angles

    @angles.setter
    def angles(self, angles):
        if self._angles == tuple(angles):
            return
        self._angles = tuple(angles)
        self.events.angles()
