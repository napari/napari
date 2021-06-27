from enum import auto

from ...utils.misc import StringEnum


class TextMode(StringEnum):
    """
    TextMode: Text setting mode.

    NONE (default mode) no text is displayed

    PROPERTY the text value is a property value

    FORMATTED allows text to be set with an f-string like syntax
    """

    NONE = auto()
    PROPERTY = auto()
    FORMATTED = auto()


class Anchor(StringEnum):
    """
    Anchor: The anchor position for text

    CENTER The text origin is centered on the layer item bounding box.

    UPPER_LEFT The text origin is on the upper left corner of the bounding box
    UPPER_RIGHT The text origin is on the upper right corner of the bounding box
    LOWER_LEFT The text origin is on the lower left corner of the bounding box
    LOWER_RIGHT The text origin is on the lower right corner of the bounding box
    """

    CENTER = auto()
    UPPER_LEFT = auto()
    UPPER_RIGHT = auto()
    LOWER_LEFT = auto()
    LOWER_RIGHT = auto()
