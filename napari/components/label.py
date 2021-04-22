"""Text label model"""
from ..utils.events import EventedModel
from ._viewer_constants import Position


class Label(EventedModel):
    """Label model"""

    # fields
    visible: bool = False
    color: str = "#FFFFFF"
    font_size: int = 10
    position: Position = Position.TOP_LEFT
    text: str = ""
