from napari.utils.compat import StrEnum


class ColorMode(StrEnum):
    """
    ColorMode: Color setting mode.
    DIRECT (default mode) allows each point to be set arbitrarily
    CYCLE allows the color to be set via a color cycle over an attribute
    COLORMAP allows color to be set via a color map over an attribute
    """

    DIRECT = 'direct'
    CYCLE = 'cycle'
    COLORMAP = 'colormap'
