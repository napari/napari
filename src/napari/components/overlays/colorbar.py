from napari.components._viewer_constants import CanvasPosition
from napari.components.overlays.base import CanvasOverlay
from napari.utils.color import ColorValue


class ColorBarOverlay(CanvasOverlay):
    """ColorBar legend overlay.

    Attributes
    ----------
    color : np.ndarray
        A (4,) color array of the text.
    size: : 2-tuple of ints
        Size of the colorbar rectangle (X, Y).
    tick_length : float
        Length of the tick marks.
    font_size : float, optional
        The font size (in points) of the text.
    position : CanvasPosition
        The position of the overlay in the canvas.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    color: ColorValue | None = None
    size: tuple[float, float] = 25, 150
    tick_length: float = 5
    font_size: float | None = None
    position: CanvasPosition = CanvasPosition.TOP_RIGHT
