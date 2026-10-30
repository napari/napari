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


SYMBOL_DICT: dict[str | Symbol, Symbol] = {x: x for x in Symbol}
SYMBOL_DICT.update({str(x): x for x in Symbol})
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

        * NONE: ignore slice thickness, only using the Dims `point`
        * ALL: project all points in the slice onto displayed dimensions
        * RESCALE_LINEAR: like ALL, but points are resized linearly based on their distance from the
            `point` of the Dims slice (size is zero at the edge of the margin)
        * RESCALE_SPHERICAL: points whose "sperical extent" intersects the `point` of the Dims slice are
            displayed, resized to match the size of the disc formed by that intersection.
            Slice margins are ignored.
        * RESCALE_SPHERICAL_THICK: points whose "sperical extent" intersects the the full Dims slice are
            displayed. If their coordinate is inside the thick slice, rendered at full size.
            Otherwise, resized to match the size of the disc created by the intersection with
            the Dims margin.
    """

    NONE = auto()
    ALL = auto()
    RESCALE_LINEAR = auto()
    RESCALE_SPHERICAL = auto()
    RESCALE_SPHERICAL_THICK = auto()
