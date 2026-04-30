from collections import OrderedDict
from enum import auto

from napari.utils.misc import StringEnum



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


class Mode(StringEnum):
    """
    Mode: Interactive mode. The normal, default mode is PAN_ZOOM, which
    allows for normal interactivity with the canvas.

    ADD allows points to be added by clicking

    SELECT allows the user to select points by clicking on them
    """

    PAN_ZOOM = auto()
    TRANSFORM = auto()
    ADD = auto()
    SELECT = auto()


class Symbol(StringEnum):
    """Valid symbol/marker types for the Points layer."""

    ARROW = auto()
    CLOBBER = auto()
    CROSS = auto()
    DIAMOND = auto()
    DISC = auto()
    HBAR = auto()
    RING = auto()
    SQUARE = auto()
    STAR = auto()
    TAILED_ARROW = auto()
    TRIANGLE_DOWN = auto()
    TRIANGLE_UP = auto()
    VBAR = auto()
    X = auto()


# Mapping of symbol alias names to the deduplicated name
SYMBOL_ALIAS = {
    '>': Symbol.ARROW,
    '+': Symbol.CROSS,
    'o': Symbol.DISC,
    '-': Symbol.HBAR,
    's': Symbol.SQUARE,
    '*': Symbol.STAR,
    '->': Symbol.TAILED_ARROW,
    'v': Symbol.TRIANGLE_DOWN,
    '^': Symbol.TRIANGLE_UP,
    '|': Symbol.VBAR,
}

SYMBOL_TRANSLATION = OrderedDict(
    [
        (Symbol.ARROW, 'arrow'),
        (Symbol.CLOBBER, 'clobber'),
        (Symbol.CROSS, 'cross'),
        (Symbol.DIAMOND, 'diamond'),
        (Symbol.DISC, 'disc'),
        (Symbol.HBAR, 'hbar'),
        (Symbol.RING, 'ring'),
        (Symbol.SQUARE, 'square'),
        (Symbol.STAR, 'star'),
        (Symbol.TAILED_ARROW, 'tailed arrow'),
        (Symbol.TRIANGLE_DOWN, 'triangle down'),
        (Symbol.TRIANGLE_UP, 'triangle up'),
        (Symbol.VBAR, 'vbar'),
        (Symbol.X, 'x'),
    ]
)

SYMBOL_TRANSLATION_INVERTED = {v: k for k, v in SYMBOL_TRANSLATION.items()}


SYMBOL_DICT: dict[str | Symbol, Symbol] = {x: x for x in Symbol}
SYMBOL_DICT.update({str(x): x for x in Symbol})
SYMBOL_DICT.update(SYMBOL_TRANSLATION_INVERTED)
SYMBOL_DICT.update(SYMBOL_ALIAS)


class Shading(StringEnum):
    """Shading: Shading mode for the points.

    NONE no shading is applied.
    SPHERICAL shading and depth buffer are modified to mimic a 3D object with spherical shape
    """

    NONE = auto()
    SPHERICAL = auto()


SHADING_TRANSLATION = {
    'none': Shading.NONE,
    'spherical': Shading.SPHERICAL,
}


class PointsProjectionMode(StringEnum):
    """
    Projection mode for aggregating a thick nD slice onto displayed dimensions.

        * NONE: ignore slice thickness, only using the dims point
        * ALL: project all points in the slice onto displayed dimensions
    """

    NONE = auto()
    ALL = auto()
