from enum import auto

from napari.utils.misc import StringEnum


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
