"""Scale bar model"""
from enum import Enum

from ..utils.events import EventedModel
from ._viewer_constants import Position


class Dimensions(str, Enum):
    """Dimension: Dimension of the scale bar units

    Sets the unit dimension of the scale bar:
        * none: Scale bar showing no units
        * si-length: Scale bar showing metric units (e.g. km, m, cm, mm, ...)
        * pixel-length: Scale bar showing pixel units (e.g. px, kpx, Mpx, ...)
    """

    NONE = "none"
    SI_M = "si-length-m"
    SI_UM = "si-length-um"
    IM_FT = "imperial-length-ft"
    PX = "pixel-length"


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
    fon_size : int
        Font size of the scale label
    dimension : str
        Display units in the scale bar. Must be one of `si-length` or `pixel-length`
    px_size : int
        Scaling of single pixel
    """

    visible: bool = False
    colored: bool = False
    ticks: bool = True
    position: Position = Position.BOTTOM_RIGHT
    font_size: int = 10
    dimension: Dimensions = Dimensions.PX
    px_size: int = 1
