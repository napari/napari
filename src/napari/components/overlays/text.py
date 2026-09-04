"""Text label model."""

from napari.components._viewer_constants import CanvasPosition
from napari.components.overlays.base import TiledCanvasOverlay
from napari.utils.color import ColorValue


class _BaseTextOverlay(TiledCanvasOverlay):
    """Label model to display arbitrary text in the canvas

    Attributes
    ----------
    color : np.ndarray
        A (4,) color array of the text overlay.
    font_size : float
        The font size (in points) of the text.
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

    color: ColorValue | None = None
    font_size: float = 10


class TextOverlay(_BaseTextOverlay):
    """Label model to display arbitrary text in the canvas

    Attributes
    ----------
    text : str
        Text to be displayed in the canvas.
    color : np.ndarray
        A (4,) color array of the text overlay.
    font_size : float
        The font size (in points) of the text.
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
