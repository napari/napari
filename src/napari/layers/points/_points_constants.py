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

    The '_ND' modes differ from the others because membership is decided by the
    extent of the point rather than the position of its center. In both cases
    the extent is a sphere of the point's size, and the portions of the off-slice
    axes are multiplied into a single size.

        * NONE: ignore slice thickness, only using the dims point
        * ALL: project all points in the slice onto displayed dimensions
        * RESCALE_LINEAR: like ALL, but points are resized linearly based on their distance from the
            center of the thick slice. The size falls to zero at the edge of the margin, so a point
            on the face of the slice is not drawn at all.
        * RESCALE_LINEAR_ND: the point is treated as an object with an extent the size of its
            diameter, rather than as a position. It is visible whenever that extent reaches the
            slice, and it fades linearly over its own radius as it leaves it, so a point in the
            slice keeps its full size and no slice thickness is needed for it to show up. This is
            the pre-0.9 `out_of_slice_display` rule and sizing.
        * RESCALE_SPHERICAL: like ALL, but points are resized to match the size of the disc created by
            a sphere centered on the point when intersecting the center of the thick slice
        * RESCALE_SPHERICAL_ND: as RESCALE_LINEAR_ND, but the size is the disc that the slice cuts
            out of the point's extent, so a point at the center of the slice keeps its full size.
    """

    NONE = auto()
    ALL = auto()
    RESCALE_LINEAR = auto()
    RESCALE_LINEAR_ND = auto()
    RESCALE_SPHERICAL = auto()
    RESCALE_SPHERICAL_ND = auto()


ND_PROJECTION_MODES = (
    PointsProjectionMode.RESCALE_LINEAR_ND,
    PointsProjectionMode.RESCALE_SPHERICAL_ND,
)
