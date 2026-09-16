from napari.components.overlays.base import SceneOverlay
from napari.utils.color import ColorValue


class CrosshairOverlay(SceneOverlay):
    """
    Overlay that displays where the cursor is located in the world.

    Attributes
    ----------
    color : ColorValue
        Color of the crosshair lines.
    gap : float
        Size of the gap in the center of the crosshair, as fraction of the canvas.
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

    color: ColorValue = ColorValue('red')
    gap: float = 0.05
