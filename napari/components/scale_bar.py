"""Scale bar model."""
from typing import Optional

from napari.utils.color import ColorValue

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
        background or the background box.
    color : ColorValue
        Scalebar and text color.
        See ``ColorValue.validate`` for supported values.
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
    box : bool
        If background box is visible or not.
    box_color : Optional[str | array-like]
        Background box color.
        See ``ColorValue.validate`` for supported values.
    unit : Optional[str]
        Unit to be used by the scale bar. The value can be set
        to `None` to display no units.
    """

    visible: bool = False
    colored: bool = False
    color: ColorValue = [1, 0, 1, 1]
    ticks: bool = True
    position: Position = Position.BOTTOM_RIGHT
    font_size: float = 10
    box: bool = False
    box_color: ColorValue = [0, 0, 0, 0.6]
    unit: Optional[str] = None
