from enum import Enum


class Position(str, Enum):
    """Position: Position on an object in the canvas.

    Sets the position of an object (e.g. scale bar) in the canvas
            * top_left: Top left of the canvas
            * top_right: Top right of the canvas
            * bottom_right: Bottom right of the canvas
            * bottom_left: Bottom left of the canvas
    """

    TOP_LEFT = 'top_left'
    TOP_RIGHT = 'top_right'
    BOTTOM_RIGHT = 'bottom_right'
    BOTTOM_LEFT = 'bottom_left'


class CursorStyle(str, Enum):
    """CursorStyle: Style on the cursor.

    Sets the style of the cursor
            * square: A square
            * circle: A circle
            * cross: A cross
            * forbidden: A forbidden symbol
            * pointing: A finger for pointing
            * standard: The standard cursor
    """

    SQUARE = 'square'
    CIRCLE = 'circle'
    CROSS = 'cross'
    FORBIDDEN = 'forbidden'
    POINTING = 'pointing'
    STANDARD = 'standard'
