"""Text label model."""

from napari.components._viewer_constants import CanvasPosition
from napari.components.overlays.base import CanvasOverlay
from napari.utils.color import ColorValue


class _BaseTextOverlay(CanvasOverlay):
    """Label model to display arbitrary text in the canvas

    Attributes
    ----------
    color : np.ndarray
        A (4,) color array of the text overlay.
    font_size : float
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
    font_size: float = 10


class TextOverlay(_BaseTextOverlay):
    """Label model to display arbitrary text in the canvas

    Attributes
    ----------
    color : np.ndarray
        A (4,) color array of the text overlay.
    font_size : float
        The font size (in points) of the text.
    text : str
        Text to be displayed in the canvas.
    position : CanvasPosition
        The position of the overlay in the canvas.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    text: str = ''


class LayerNameOverlay(_BaseTextOverlay):
    """Label model to display layer name text in the canvas

    Attributes
    ----------
    color : np.ndarray
        A (4,) color array of the text overlay.
    font_size : float
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

    position: CanvasPosition = CanvasPosition.TOP_LEFT


class CurrentSliceOverlay(_BaseTextOverlay):
    """Label model to display the current dims slice in the canvas

    Attributes
    ----------
    color : np.ndarray
        A (4,) color array of the text overlay.
    font_size : float
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
