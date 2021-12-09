from pydantic import Field

from ...utils.events import EventedModel

# from .axes import Axes
from .interaction_box import InteractionBox
from .scale_bar import ScaleBar
from .text import Text


class Overlays(EventedModel):
    """A collection of components that will draw on top of layer data.

    Attributes
    ----------
    interaction_box : InteractionBox
        A box that can be used to select and transform layers or data within a layer.
    axes:
        Axes indicating world coordinate origin and orientation.
    text:
        Text overlay with arbitrary information.
    scale_bar:
        Scale bar indicating size in world coordinates.
    visible:
        Whether the overlays are visible.
    """

    # fields
    interaction_box: InteractionBox = Field(
        default_factory=InteractionBox, allow_mutation=False
    )
    # axes: Axes = Field(
    # default_factory=Axes, allow_mutation=False
    # )
    text: Text = Field(default_factory=Text, allow_mutation=False)
    scale_bar: ScaleBar = Field(default_factory=ScaleBar, allow_mutation=False)
    visible: bool = True
