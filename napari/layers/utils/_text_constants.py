from enum import auto

from ...utils.misc import StringEnum


class TextMode(StringEnum):
    """
    Text: Text setting mode.

    NONE (default mode) no text is displayed

    PROPERTY the text value is a property value

    FORMATTED allows text to be set with an f string like syntax
    """

    NONE = auto()
    PROPERTY = auto()
    FORMATTED = auto()
