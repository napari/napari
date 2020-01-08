from ..utils.event import EmitterGroup


class Camera:
    """Camera object modeling position and view of the camera.

    Parameters
    ----------
    center : 2-tuple or 3-tuple
        Center point of camera view for 2D or 3D viewing.
    scale : float
        Zoom level.
    angles : 3-tuple
        Euler angles of camera in 3D viewing (rx, ry, rz), in degrees.
    """

    def __init__(self, *, center=(0, 0), scale=512, angles=(0, 0, 90)):

        self._center = center
        self._scale = scale
        self._angles = angles

        self.events = EmitterGroup(
            source=self, auto_connect=True, update=None, ndisplay=None,
        )

    @property
    def center(self):
        """tuple: Center point of camera view for 2D or 3D viewing."""
        return self._center

    @center.setter
    def center(self, center):
        if self._center == tuple(center):
            return
        ndisplay = len(center)
        if ndisplay not in [2, 3]:
            raise ValueError(
                f'Center must be length 2 or 3, got length {ndisplay}.'
            )
        old_ndisplay = self.ndisplay
        self._center = tuple(center)
        if old_ndisplay != ndisplay:
            self.events.ndisplay()
        self.events.update()

    @property
    def ndisplay(self):
        """int: Dimensionality of the camera rendering."""
        return len(self.center)

    @property
    def scale(self):
        """float: Zoom level."""
        return self._scale

    @scale.setter
    def scale(self, scale):
        if self._scale == scale:
            return
        self._scale = scale
        self.events.update()

    @property
    def angles(self):
        """3-tuple: Euler angles of camera in 3D viewing, in degrees."""
        return self._angles

    @angles.setter
    def angles(self, angles):
        if self._angles == angles:
            return
        self._angles = angles
        self.events.update()

    def update(self, center=None, scale=None, angles=None):
        """Update camera position to new values.

        Parameters
        ----------
        center : tuple
            Center point of camera view for 2D or 3D viewing, must be length 2
            or 3.
        scale : float
            Zoom level.
        angles : 3-tuple
            Euler angles of camera in 3D viewing (rx, ry, rz), in degrees.
        """
        if center is not None:
            self.center = center
        if scale is not None:
            self.scale = scale
        if angles is not None:
            self.angles = angles

    def reset(self):
        """Reset camera position to initial values."""
        self.update(center=(0,) * self.ndisplay, scale=512, angles=(0, 0, 90))
