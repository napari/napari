from enum import auto

from napari.utils.misc import StringEnum


class ImageProjectionMode(StringEnum):
    """
    Projection mode for aggregating a thick nD slice onto displayed dimensions.

        * NONE: ignore slice thickness, only using the dims point
        * SUM: sum data across the thick slice
        * MEAN: average data across the thick slice
        * MAX: display the maximum value across the thick slice
        * MIN: display the minimum value across the thick slice
    """

    NONE = auto()
    SUM = auto()
    MEAN = auto()
    MAX = auto()
    MIN = auto()
