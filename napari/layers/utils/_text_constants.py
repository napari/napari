from enum import auto

from ...utils.misc import StringEnum


class TextMode(StringEnum):
    """
    TextMode: Text setting mode.

    NONE (default mode) no text is displayed

    PROPERTY the text value is a property value

    FORMATTED allows text to be set with an f string like syntax
    """

    NONE = auto()
    PROPERTY = auto()
    FORMATTED = auto()


class Anchor(StringEnum):
    """
    Anchor: The anchor position for text

    CENTER The text origin is centered on the layer item bounding box.
    """

    CENTER = auto()
