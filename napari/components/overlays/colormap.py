from napari.components.overlays.base import CanvasOverlay


class ColormapOverlay(CanvasOverlay):
    """Colormap legend overlay.

    Attributes
    ----------
    position : CanvasPosition
        The position of the overlay in the canvas.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """
