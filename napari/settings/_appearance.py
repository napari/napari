from pydantic import Field

from ..utils.events.evented_model import EventedModel
from ..utils.translations import trans
from ._fields import Theme


class AppearanceSettings(EventedModel):
    theme: Theme = Field(
        "dark",
        title=trans._("Theme"),
        description=trans._("Select the user interface theme."),
        env="napari_theme",
    )
    highlight_thickness: int = Field(
        1,
        title=trans._("Highlight thickness"),
        description=trans._(
            "Select the highlight thickness when hovering over shapes/points."
        ),
        ge=1,
        le=10,
    )
    layer_tooltip_visibility: bool = Field(
        False,
        title=trans._("Show layer tooltips"),
        description=trans._("Toggle to display a tooltip on mouse hover."),
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']
