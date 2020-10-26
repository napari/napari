from ..utils.events import EmitterGroup
from ._viewer_constants import CursorStyle


class Cursor:
    """Cursor object with position and properties of the cursor.

    Attributes
    ----------
    position : tuple or None
        Position of the cursor in world coordinates. None if outside the
        world.
    scaled : bool
        Flag to indicate whether cursor size should be scaled to zoom.
        Only relevant for circle and square cursors which are drawn
        with a particular size.
    style : str
        Style of the cursor. Must be one of
            * square: A square
            * circle: A circle
            * cross: A cross
            * forbidden: A forbidden symbol
            * pointing: A finger for pointing
            * standard: The standard cursor
    size : float
        Size of the cursor in canvas pixels.Only relevant for circle
        and square cursors which are drawn with a particular size.
    """

    def __init__(self):

        self._position = None
        self.scaled = True
        self._size = 1
        self._style = CursorStyle('standard')

        self.events = EmitterGroup(
            source=self,
            auto_connect=True,
            position=None,
            style=None,
            size=None,
        )

    @property
    def position(self):
        """tuple: Position of the cursor in world coordinates."""
        return self._position

    @position.setter
    def position(self, position):
        if self._position == tuple(position):
            return
        self._position = tuple(position)
        self.events.position()

    @property
    def size(self):
        """int: Size of the cursor in canvas pixels."""
        return self._size

    @size.setter
    def size(self, size):
        if self._size == size:
            return
        self._size = size
        self.events.size()

    @property
    def style(self):
        """str: Style of the cursor."""
        return str(self._style)

    @style.setter
    def style(self, style):
        if self._style == CursorStyle(style):
            return
        self._style = CursorStyle(style)
        self.events.style()
