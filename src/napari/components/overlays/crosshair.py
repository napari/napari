from pydantic import Field, PositiveInt

from napari.components.overlays.base import SceneOverlay
from napari.utils.color import ColorValue


class CrosshairOverlay(SceneOverlay):
    """
    Overlay that displays where the cursor is located in the world.

    Attributes
    ----------
    color : ColorValue or 3-tuple of ColorValue or None
        Color of the crosshair lines. A single color is used for all lines.
        Three colors are used for each displayed axis, in the same order
        as `viewer.dims.displayed` (in 2D, only the last two are used).
        If None, each line is colored like the matching axis of the axes
        overlay.
    gap : int
        Diameter of the gap in the center of the crosshair, in canvas pixels.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    blending : Blending
        One of a list of preset blending modes that determines how RGB and
        alpha values of the overlay get mixed with the visuals below.
    """

    # left_to_right so that 3 colors are not parsed as a single ColorValue
    color: tuple[ColorValue, ColorValue, ColorValue] | ColorValue | None = (
        Field(default=None, union_mode='left_to_right')
    )
    gap: PositiveInt = 20
    opacity: float = 0.7
