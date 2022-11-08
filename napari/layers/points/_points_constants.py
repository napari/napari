from collections import OrderedDict
from enum import auto

from ...utils.misc import StringEnum
from ...utils.translations import trans


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

    ADD = auto()
    SELECT = auto()
    PAN_ZOOM = auto()
    TRANSFORM = auto()


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
    'o': Symbol.DISC,
    '*': Symbol.STAR,
    '+': Symbol.CROSS,
    '-': Symbol.HBAR,
    '->': Symbol.TAILED_ARROW,
    '>': Symbol.ARROW,
    '^': Symbol.TRIANGLE_UP,
    'v': Symbol.TRIANGLE_DOWN,
    's': Symbol.SQUARE,
    '|': Symbol.VBAR,
}

SYMBOL_STR_SIZE = max(len(s.value) for s in Symbol)

SYMBOL_TRANSLATION = OrderedDict(
    [
        (Symbol.ARROW, trans._('arrow')),
        (Symbol.CLOBBER, trans._('clobber')),
        (Symbol.CROSS, trans._('cross')),
        (Symbol.DIAMOND, trans._('diamond')),
        (Symbol.DISC, trans._('disc')),
        (Symbol.HBAR, trans._('hbar')),
        (Symbol.RING, trans._('ring')),
        (Symbol.SQUARE, trans._('square')),
        (Symbol.STAR, trans._('star')),
        (Symbol.TAILED_ARROW, trans._('tailed arrow')),
        (Symbol.TRIANGLE_DOWN, trans._('triangle down')),
        (Symbol.TRIANGLE_UP, trans._('triangle up')),
        (Symbol.VBAR, trans._('vbar')),
        (Symbol.X, trans._('x')),
    ]
)


class Shading(StringEnum):
    """Shading: Shading mode for the points.

    NONE no shading is applied.
    SPHERICAL shading and depth buffer are modified to mimic a 3D object with spherical shape
    """

    NONE = auto()
    SPHERICAL = auto()


SHADING_TRANSLATION = {
    trans._("none"): Shading.NONE,
    trans._("spherical"): Shading.SPHERICAL,
}
