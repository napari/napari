from collections import OrderedDict
from enum import auto

from napari.utils.misc import StringEnum
from napari.utils.translations import trans


class VectorStyle(StringEnum):
    """STYLE: Display style for the vectors.

    Selects a preset display style in that determines how
            vectors are displayed.
            VectorStyle.LINE
                Displays vectors as rectangular lines.
            VectorStyle.TRIANGLE
                Displays vectors as triangles.
            VectorStyle.ARROW
                Displays vectors as arrows.
    """

    LINE = auto()
    TRIANGLE = auto()
    ARROW = auto()


VECTORSTYLE_TRANSLATIONS = OrderedDict(
    [
        (VectorStyle.LINE, trans._("line")),
        (VectorStyle.TRIANGLE, trans._("triangle")),
        (VectorStyle.ARROW, trans._("arrow")),
    ]
)
