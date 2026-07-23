from napari.components._viewer_constants import CanvasPosition
from napari.components.overlays.base import CanvasOverlay
from napari.layers.base._base_constants import Blending


class FloatingAxesOverlay(CanvasOverlay):
    """Axes indicating camera orientation, pinned to a canvas corner.

    Attributes
    ----------
    labels : bool
        If axis labels ('Z', 'Y', 'X') are visible or not.
    colored : bool
        If axes are colored or not. If colored then default
        coloring is z=magenta, y=yellow, x=cyan. If not
        colored than axes are the color opposite of
        the canvas background.
    dashed : bool
        If axes are dashed or not. If not dashed then
        all the axes are solid. If dashed then x=solid,
        y=dashed, z=dotted.
    arrows : bool
        If axes have arrowheads or not.
    size : float
        Size taken up by the overlay in canvas pixels.
    position : CanvasPosition
        The position of the overlay in the canvas.
    box : bool
        Whether the background box is visible or not.
    box_color : ColorValue or None
        Background box color. If unset, it defaults to the canvas color.
    gridded : bool
        The overlay will be duplicated across all grid cells in gridded mode.
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

    labels: bool = True
    colored: bool = True
    dashed: bool = False
    arrows: bool = True
    size: float = 100
    blending: Blending = Blending.TRANSLUCENT_NO_DEPTH
    position: CanvasPosition = CanvasPosition.BOTTOM_LEFT
