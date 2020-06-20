from enum import auto

from ...utils.misc import StringEnum


class TextMode(StringEnum):
    """
    Text: Text setting mode.

    DIRECT (default mode) allows each text to be set arbitrarily

    PROPERTY the text value is a property value

    FORMATTED allows text to be set with an f string like syntax
    """

    DIRECT = auto()
    PROPERTY = auto()
    FORMATTED = auto()
