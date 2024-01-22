"""Text label model."""
from napari._pydantic_compat import Field
from napari.components.overlays.base import CanvasOverlay
from napari.utils.color import ColorValue


class TextOverlay(CanvasOverlay):
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

    color: ColorValue = Field(
        default_factory=lambda: ColorValue((0.5, 0.5, 0.5, 1.0))
    )
    font_size: float = 10
    text: str = ""
