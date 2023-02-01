from collections import OrderedDict
from enum import IntEnum, auto

from napari.utils.misc import StringEnum
from napari.utils.translations import trans


class Blending(StringEnum):
    """BLENDING: Blending mode for the layer.

    Selects a preset blending mode in vispy that determines how
            RGB and alpha values get mixed.
            Blending.OPAQUE
                Allows for only the top layer to be visible and corresponds to
                depth_test=True, cull_face=False, blend=False.
            Blending.TRANSLUCENT
                Allows for multiple layers to be blended with different opacity
                and corresponds to depth_test=True, cull_face=False,
                blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'),
                and blend_equation=('func_add').
            Blending.TRANSLUCENT_NO_DEPTH
                Allows for multiple layers to be blended with different opacity,
                but no depth testing is performed.
                and corresponds to depth_test=False, cull_face=False,
                blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'),
                and blend_equation=('func_add').
            Blending.ADDITIVE
                Allows for multiple layers to be blended together with
                different colors and opacity. Useful for creating overlays. It
                corresponds to depth_test=False, cull_face=False, blend=True,
                blend_func=('src_alpha', 'one').
            Blending.MINIMUM
                Allows for multiple layers to be blended together such that
                the minimum of each color and alpha are selected.
                Useful for creating overlays with inverted colormaps. It
                corresponds to depth_test=False, cull_face=False, blend=True,
                blend_equation='min'.
    """

    TRANSLUCENT = auto()
    TRANSLUCENT_NO_DEPTH = auto()
    ADDITIVE = auto()
    MINIMUM = auto()
    OPAQUE = auto()


BLENDING_TRANSLATIONS = OrderedDict(
    [
        (Blending.TRANSLUCENT, trans._("translucent")),
        (Blending.TRANSLUCENT_NO_DEPTH, trans._("translucent_no_depth")),
        (Blending.ADDITIVE, trans._("additive")),
        (Blending.MINIMUM, trans._("minimum")),
        (Blending.OPAQUE, trans._("opaque")),
    ]
)


class Mode(StringEnum):
    """
    Mode: Interactive mode. The normal, default mode is PAN_ZOOM, which
    allows for normal interactivity with the canvas.

    TRANSFORM allows for manipulation of the layer transform.
    """

    PAN_ZOOM = auto()
    TRANSFORM = auto()


class InteractionBoxHandle(IntEnum):
    """
    Handle indices for the InteractionBox overlay.

    Vertices are generated according to the following scheme:
        8
        |
    0---4---2
    |       |
    5   9   6
    |       |
    1---7---3

    Note that y is actually upside down in the canvas in vispy coordinates.
    """

    TOP_LEFT = 0
    TOP_CENTER = 4
    TOP_RIGHT = 2
    CENTER_LEFT = 5
    CENTER_RIGHT = 6
    BOTTOM_LEFT = 1
    BOTTOM_CENTER = 7
    BOTTOM_RIGHT = 3
    ROTATION = 8
    INSIDE = 9

    @classmethod
    def opposite_handle(cls, handle):
        opposites = {
            InteractionBoxHandle.TOP_LEFT: InteractionBoxHandle.BOTTOM_RIGHT,
            InteractionBoxHandle.TOP_CENTER: InteractionBoxHandle.BOTTOM_CENTER,
            InteractionBoxHandle.TOP_RIGHT: InteractionBoxHandle.BOTTOM_LEFT,
            InteractionBoxHandle.CENTER_LEFT: InteractionBoxHandle.CENTER_RIGHT,
        }

        opposites.update({v: k for k, v in opposites.items()})
        if (opposite := opposites.get(handle, None)) is None:
            raise ValueError(f'{handle} has no opposite handle.')
        return opposite

    @classmethod
    def corners(cls):
        return (
            cls.TOP_LEFT,
            cls.TOP_RIGHT,
            cls.BOTTOM_LEFT,
            cls.BOTTOM_RIGHT,
        )
