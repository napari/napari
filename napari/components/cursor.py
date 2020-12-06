from typing import Tuple

from ..utils.events.dataclass import Property, evented_dataclass
from ._viewer_constants import CursorStyle, CursorType


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
    is_dragging : bool
        If cursor is currently dragging.
    type : str
        Type of last cursor interaction.
            * MOUSE_MOVE: A mouse move event
            * MOUSE_PRESS: A mouse press event
            * MOUSE_RELEASE: A mouse release event
            * MOUSE_WHEEL: A mouse wheel event
    """

    position: Property[Tuple, None, tuple] = ()
    canvas_position: Property[Tuple, None, tuple] = (0, 0)
    scaled: bool = True
    size: int = 1
    style: Property[CursorStyle, str, CursorStyle] = CursorStyle.STANDARD
    is_dragging: bool = False
    modifiers: Property[Tuple, None, tuple] = ()
    type: Property[CursorType, str, CursorType] = CursorType.MOUSE_MOVE
    inverted: bool = False
    delta: Property[Tuple, None, tuple] = (0, 0)
