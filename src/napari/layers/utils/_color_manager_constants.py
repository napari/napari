from enum import StrEnum, auto


class ColorMode(StrEnum):
    """
    ColorMode: Color setting mode.
    - direct: (default mode) allows each point to be set arbitrarily
    - cycle: allows the color to be set via a color cycle over an attribute
    - colormap: allows color to be set via a color map over an attribute
    """

    direct = auto()
    cycle = auto()
    colormap = auto()
