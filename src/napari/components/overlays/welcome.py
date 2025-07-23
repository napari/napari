from typing import Literal

from napari.components.overlays.base import CanvasOverlay


class WelcomeOverlay(CanvasOverlay):
    """Welcome screen overlay."""

    visible: bool = True
    # not settable in this specific overlay
    position: None = None
    gridded: Literal[False] = False
