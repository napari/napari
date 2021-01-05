from typing import Tuple

from ..utils.events.dataclass import Property, evented_dataclass
from ._viewer_constants import CursorStyle


@evented_dataclass
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
    size : float
        Size of the cursor in canvas pixels.Only relevant for circle
        and square cursors which are drawn with a particular size.
    style : str
        Style of the cursor. Must be one of
            * square: A square
            * circle: A circle
            * cross: A cross
            * forbidden: A forbidden symbol
            * pointing: A finger for pointing
            * standard: The standard cursor
    """

    position: Property[Tuple, None, tuple] = ()
    scaled: bool = True
    size: int = 1
    style: Property[CursorStyle, str, CursorStyle] = CursorStyle.STANDARD
