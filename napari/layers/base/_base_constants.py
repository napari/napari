from collections import OrderedDict
from enum import auto

from ...utils.misc import StringEnum
from ...utils.translations import trans


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
                blend=True, blend_func=('src_alpha', 'one_minus_src_alpha').
            Blending.ADDITIVE
                Allows for multiple layers to be blended together with
                different colors and opacity. Useful for creating overlays. It
                corresponds to depth_test=False, cull_face=False, blend=True,
                blend_func=('src_alpha', 'one').
    """

    TRANSLUCENT = auto()
    ADDITIVE = auto()
    OPAQUE = auto()


BLENDING_TRANSLATIONS = OrderedDict(
    [
        (Blending.TRANSLUCENT, trans._("translucent")),
        (Blending.ADDITIVE, trans._("additive")),
        (Blending.OPAQUE, trans._("opaque")),
    ]
)
