from enum import Enum
import sys


class Mode(Enum):
    """MODE: Interactive mode. The normal, default mode is PAN_ZOOM, which
    allows for normal interactivity with the canvas.

    The SELECT mode allows for entire shapes to be selected, moved and
    resized.

    The DIRECT mode allows for shapes to be selected and their individual
    vertices to be moved.

    The VERTEX_INSERT and VERTEX_REMOVE modes allow for individual
    vertices either to be added to or removed from shapes that are already
    selected. Note that shapes cannot be selected in this mode.

    The ADD_RECTANGLE, ADD_ELLIPSE, ADD_LINE, ADD_PATH, and ADD_POLYGON
    modes all allow for their corresponding shape type to be added.
    """
    PAN_ZOOM = 0
    SELECT = 1
    DIRECT = 2
    ADD_RECTANGLE = 3
    ADD_ELLIPSE = 4
    ADD_LINE = 5
    ADD_PATH = 6
    ADD_POLYGON = 7
    VERTEX_INSERT = 8
    VERTEX_REMOVE = 9


BOX_WITH_HANDLE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
BOX_LINE_HANDLE = [1, 2, 4, 6, 0, 1, 8]
BOX_LINE = [0, 2, 4, 6, 0]
BOX_TOP_LEFT = 0
BOX_TOP_CENTER = 1
BOX_BOTTOM_RIGHT = 4
BOX_BOTTOM_LEFT = 6
BOX_CENTER = 8
BOX_HANDLE = 9
BOX_LEN = 8

BACKSPACE = 'delete' if sys.platform == 'darwin' else 'backspace'
