"""Text label model"""
from ..utils.events import EventedModel
from ._viewer_constants import TextOverlayPosition


class TextOverlay(EventedModel):
    """Label model to display arbitrary text in the canvas

    Attributes
    ----------
    visible : bool
        If label is visible or not
    color : str
        Color (hex) of the label
    font_size : int
        Size of the font
    position : str
        Position of the label in the canvas. Must be one of
        'top left', 'top right', 'top center', 'bottom right',
        'bottom left', 'bottom_center'.
        Default value is 'top left'
    text : str
        Text to be displayed in the canvas
    """

    # fields
    visible: bool = False
    color: str = "#FFFFFF"
    font_size: int = 10
    position: TextOverlayPosition = TextOverlayPosition.TOP_LEFT
    text: str = ""
