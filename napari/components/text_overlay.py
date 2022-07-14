"""Text label model."""
from napari.utils.color import ColorValue

from ..utils.events import EventedModel
from ._viewer_constants import TextOverlayPosition


class TextOverlay(EventedModel):
    """Label model to display arbitrary text in the canvas

    Attributes
    ----------
    visible : bool
        If text overlay is visible or not.
    color : np.ndarray
        A (4,) color array of the text overlay.
    font_size : float
        The font size (in points) of the text.
    position : str
        Position of the text overlay in the canvas. Must be one
         of 'top left', 'top right', 'top center', 'bottom right',
        'bottom left', 'bottom_center'.
        Default value is 'top left'
    text : str
        Text to be displayed in the canvas.
    """

    # fields
    visible: bool = False
    color: ColorValue = (0.5, 0.5, 0.5, 1.0)
    font_size: float = 10
    position: TextOverlayPosition = TextOverlayPosition.TOP_LEFT
    text: str = ""
