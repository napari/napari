from enum import auto

import numpy as np

from ...utils.misc import StringEnum


class ColorMode(StringEnum):
    """
    ColorMode: Color setting mode.

    DIRECT (default mode) allows each point to be set arbitrarily

    CYCLE allows the color to be set via a color cycle over an attribute

    COLORMAP allows color to be set via a color map over an attribute
    """

    DIRECT = auto()
    CYCLE = auto()
    COLORMAP = auto()


DEFAULT_COLOR_CYCLE = np.array([[1, 0, 1, 1], [0, 1, 0, 1]])
