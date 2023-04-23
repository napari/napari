from typing import Tuple

from napari.components.overlays.base import CanvasOverlay


class BrushCircleOverlay(CanvasOverlay):
    size: int = 10
    position: Tuple[int, int] = (0, 0)
