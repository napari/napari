from napari.components.overlays.base import CanvasOverlay


class BrushCircleOverlay(CanvasOverlay):
    """
    Overlay that displays a circle for a brush on a canvas.

    Attributes
    ----------
    size : int
        The diameter of the brush circle in canvas pixels.
    position : Tuple[int, int]
        The position (x, y) of the center of the brush circle on the canvas.
    position_is_frozen : bool
        If True, the overlay does not respond to mouse movements.
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

    size: int = 10
    position: tuple[int, int] = (0, 0)
    position_is_frozen: bool = False
