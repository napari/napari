from typing import Tuple

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
    """

    size: int = 10
    position: Tuple[int, int] = (0, 0)
    position_is_frozen: bool = False
