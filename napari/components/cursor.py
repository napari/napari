from ..utils.events import EmitterGroup


class Cursor:
    """Cursor object with position and properties of the cursor.

    Parameters
    ----------
    camera : napari.components.Camera
        Camera model.
    dims : napari.components.Dims
        Dims model of the viewer.

    Attributes
    ----------
    canvas : 2-tuple or None
        Position of the cursor in the canvas, relative to top-left corner.
        None if outside the canvas.
    world : tuple or None
        Position of the cursor in world coordinates. None if outside the
        world.
    style : str
        Style of the cursor. Muse be one of ....
    size : float
        Size of the cursor in canvas pixels.
    """

    def __init__(self, camera, dims):
        self._camera = camera
        self._dims = dims

        self._canvas = None

        self.events = EmitterGroup(
            source=self, auto_connect=True, canvas=None, style=None, size=None,
        )

    @property
    def canvas(self):
        """tuple: Center point of camera view for 2D or 3D viewing."""
        return self._canvas

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

    # @property
    # def ndisplay(self):
    #     """int: Dimensionality of the camera rendering."""
    #     return self._dims.ndisplay

    # @ndisplay.setter
    # def ndisplay(self, ndisplay):
    #     self._dims.ndisplay = ndisplay

    # @property
    # def zoom(self):
    #     """float: Scale from canvas pixels to world pixels."""
    #     return self._zoom

    # @zoom.setter
    # def zoom(self, zoom):
    #     if self._zoom == zoom:
    #         return
    #     self._zoom = zoom
    #     self.events.zoom()

    # @property
    # def angles(self):
    #     """3-tuple: Euler angles of camera in 3D viewing, in degrees."""
    #     if self.ndisplay == 3:
    #         return self._angles
    #     else:
    #         return (0, 0, 90)

    # @angles.setter
    # def angles(self, angles):
    #     if self._angles == tuple(angles):
    #         return
    #     self._angles = tuple(angles)
    #     self.events.angles()
