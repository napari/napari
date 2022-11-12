from pydantic import Field

from napari.components.overlays.interaction_box import InteractionBox
from napari.utils.events import EventedModel


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
