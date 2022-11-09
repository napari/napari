from enum import auto

from napari.utils.misc import StringEnum
from napari.utils.translations import trans


class Shading(StringEnum):
    """Shading: Shading mode for the surface.

    Selects a preset shading mode in vispy that determines how
    color is computed in the scene.
    See also: https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glShadeModel.xml
            Shading.NONE
                Computed color is interpreted as input color, unaffected by
                lighting. Corresponds to shading='none'.
            Shading.FLAT
                Computed colours are the color at a specific vertex for each
                primitive in the mesh. Corresponds to shading='flat'.
            Shading.SMOOTH
                Computed colors are interpolated between vertices for each
                primitive in the mesh. Corresponds to shading='smooth'
    """

    NONE = auto()
    FLAT = auto()
    SMOOTH = auto()


SHADING_TRANSLATION = {
    trans._("none"): Shading.NONE,
    trans._("flat"): Shading.FLAT,
    trans._("smooth"): Shading.SMOOTH,
}
