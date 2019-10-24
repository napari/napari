from enum import Enum, auto

from ...util.misc import StringEnum


class Mode(StringEnum):
    """
    Mode: Interactive mode. The normal, default mode is PAN_ZOOM, which
    allows for normal interactivity with the canvas.

    ADD allows points to be added by clicking

    SELECT allows the user to select points by clicking on them
    """

    ADD = auto()
    SELECT = auto()
    PAN_ZOOM = auto()
