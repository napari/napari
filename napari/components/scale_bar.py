"""Scale bar model."""
from typing import Optional, Union
from pydantic import validator

from ..utils.colormaps.standardize_color import transform_color
from ..utils.events import EventedModel
from ..utils.events.custom_types import Array
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
    color : Optional[str | array-like]
        Scalebar and text color. Can be any color name recognized by vispy or
        hex value if starting with `#`. If array-like must be 1-dimensional
        array with 3 or 4 elements.
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
        Box color. Can be any color name recognized by vispy or
        hex value if starting with `#`. If array-like must be 1-dimensional
        array with 3 or 4 elements.
    unit : Optional[str]
        Unit to be used by the scale bar. The value can be set
        to `None` to display no units.
    """

    visible: bool = False
    colored: bool = False
    color: Optional[Array[float, (4,)]] = None
    ticks: bool = True
    position: Position = Position.BOTTOM_RIGHT
    font_size: float = 10
    box: bool = False
    box_color: Optional[Array[float, (4,)]] = None
    unit: Optional[str] = None

    @validator('color', pre=True)
    def _coerce_color(cls, v):
        return transform_color(v)[0]

    @validator('box_color', pre=True)
    def _coerce_box_color(cls, v):
        return transform_color(v)[0]