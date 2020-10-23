from enum import auto

from ..utils.misc import StringEnum


class Position(StringEnum):
    """Position: Position on an object in the canvas.

    Sets the position of an object (e.g. scale bar) in the canvas
            * top_left: Top left of the canvas
            * top_right: Top right of the canvas
            * bottom_right: Bottom right of the canvas
            * bottom_left: Bottom left of the canvas
    """

    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_RIGHT = auto()
    BOTTOM_LEFT = auto()


class CursorStyle(StringEnum):
    """CursorStyle: Style on the cursor.

    Sets the style of the cursor
            * square: A square
            * circle: A circle
            * cross: A cross
            * forbidden: A forbidden symbol
            * pointing: A finger for pointing
            * standard: The standard cursor
    """

    SQUARE = auto()
    CIRCLE = auto()
    CROSS = auto()
    FORBIDDEN = auto()
    POINTING = auto()
    STANDARD = auto()
