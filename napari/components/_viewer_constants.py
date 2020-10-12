from enum import auto

from ..utils.misc import StringEnum


class AxesStyle(StringEnum):
    """AXESSTYLE: Style for Axes.

    Determines style of the axes for display.
        AxesStyle.COLORED
            Axes are colored with x=cyan, y=yellow, z=magenta.
        AxesStyle.DASHED
            Axes are dashed with x=solid, y=dotted, z=dashed.
    """

    COLORED = auto()
    DASHED = auto()
