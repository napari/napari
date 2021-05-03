"""Scale bar model."""
from typing import Optional

from ..utils.events import EventedModel
from ._viewer_constants import Position


class ScaleBar(EventedModel):
    """Scale bar indicating size in world coordinates.

    Attributes
    ----------
    visible : bool
        If scale bar is visible or not.
    colored : bool
        If scale bar are colored or not. If colored then
        default color is magenta. If not colored than
        scale bar color is the opposite of the canvas
        background.
    ticks : bool
        If scale bar has ticks at ends or not.
    position : str
        Position of the scale bar in the canvas. Must be one of
        'top left', 'top right', 'bottom right', 'bottom left'.
        Default value is 'bottom right'.
    background_color : np.ndarray
        Background color of canvas. If scale bar is not colored
        then it has the color opposite of this color.
    font_size : float
        The font size (in points) of the text.
    unit : Optional[str]
        Unit to be used by the scale bar. The value can be set
        to `None` to display no units.
    """

    visible: bool = False
    colored: bool = False
    ticks: bool = True
    position: Position = Position.BOTTOM_RIGHT
    font_size: float = 10
    unit: Optional[str] = None
