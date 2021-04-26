"""Text label model"""
from typing import Optional

from pydantic import validator

from ..utils.colormaps.standardize_color import transform_color
from ..utils.events import EventedModel
from ..utils.events.custom_types import Array
from ._viewer_constants import TextOverlayPosition


class TextOverlay(EventedModel):
    """Label model to display arbitrary text in the canvas

    Attributes
    ----------
    visible : bool
        If label is visible or not
    color : Optional[np.ndarray]
        A (4,) color array of the text overlay
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
    color: Optional[Array[float, (4,)]] = (1.0, 1.0, 1.0, 1.0)
    font_size: int = 10
    position: TextOverlayPosition = TextOverlayPosition.TOP_LEFT
    text: str = ""

    @validator('color', pre=True)
    def _coerce_color(cls, v):
        if v is None:
            return v
        elif len(v) == 0:
            return None
        else:
            return transform_color(v)[0]
