from ..utils.event import EmitterGroup


class Camera:
    """Camera object modeling position and view of the camera.

    Parameters
    ----------
    center : 2-tuple or 3-tuple
        Center point of camera view for 2D or 3D viewing.
    scale : float
        Zoom level.
    angle : 2-tuple
        Phi and Psi angles of camera view used in 3D viewing.
    """

    def __init__(self, *, center=(0, 0), scale=512, angle=(0, 0)):

        self._center = center
        self._scale = scale
        self._angle = angle

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
    def angle(self):
        """2-tuple: Phi and Psi angles of camera view used in 3D viewing."""
        return self._angle

    @angle.setter
    def angle(self, angle):
        if self._angle == angle:
            return
        self._angle = angle
        self.events.update()

    def update(self, center=None, scale=None, angle=None):
        """Update camera position to new values.

        Parameters
        ----------
        center : tuple
            Center point of camera view for 2D or 3D viewing, must be length 2
            or 3.
        scale : float
            Zoom level.
        angle : 2-tuple
            Phi and Psi angles of camera view used in 3D viewing.
        """
        if center is not None:
            self.center = center
        if scale is not None:
            self.scale = scale
        if angle is not None:
            self.angle = angle

    def reset(self):
        """Reset camera position to initial values."""
        self.update(center=(0,) * self.ndisplay, scale=512, angle=(0, 0))
