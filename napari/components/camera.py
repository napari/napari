from ..utils.events import EmitterGroup


class Camera:
    """Camera object modeling position and view of the camera.

    Parameters
    ----------
    center : 2-tuple or 3-tuple
        Center point of camera view for 2D or 3D viewing.
    size : float
        Max size of data to display in canvas in data units.
    angles : 3-tuple
        Euler angles of camera in 3D viewing (rx, ry, rz), in degrees.
    """

    def __init__(self, *, center=(0, 0), size=1, angles=(0, 0, 90)):

        self._center = center
        self._size = size
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
    def size(self):
        """float: Max size of data to display in canvas in data units.."""
        return self._size

    @size.setter
    def size(self, size):
        if self._size == size:
            return
        self._size = size
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

    def update(self, center=None, size=None, angles=None):
        """Update camera position to new values.

        Parameters
        ----------
        center : tuple
            Center point of camera view for 2D or 3D viewing, must be length 2
            or 3.
        size : float
            Max size of data to display in canvas in data units.
        angles : 3-tuple
            Euler angles of camera in 3D viewing (rx, ry, rz), in degrees.
        """
        if center is not None:
            self.center = center
        if size is not None:
            self.size = size
        if angles is not None:
            self.angles = angles
