from pydantic import Field, validator

from ..utils._pydantic import Array, EventedModel
from ..utils.colormaps.colormap_utils import make_default_color_array
from ..utils.colormaps.standardize_color import transform_single_color
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
    """

    visible: bool = False
    colored: bool = False
    ticks: bool = True
    position: Position = Position.BOTTOM_RIGHT
    background_color: Array[float, (4,)] = Field(
        default_factory=make_default_color_array
    )

    # validators
    _ensure_single_color = validator(
        'background_color', pre=True, allow_reuse=True
    )(transform_single_color)
