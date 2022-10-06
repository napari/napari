from pydantic import Field

from ...utils.events import EventedModel
from .interaction_box import InteractionBox


class Overlays(EventedModel):
    """A collection of components that will draw on top of layer data.

    Attributes
    ----------
    interaction_box : InteractionBox
        A box that can be used to select and transform layers or data within a layer.
    """

    # fields
    interaction_box: InteractionBox = Field(
        default_factory=InteractionBox, allow_mutation=False
    )
