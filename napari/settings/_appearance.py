from pydantic import Field

from ..utils.events.custom_types import conint
from ..utils.events.evented_model import EventedModel
from ..utils.theme import available_themes
from ..utils.translations import trans
from ._fields import Theme

HighlightThickness = conint(ge=1, le=10)


class AppearanceSettings(EventedModel):
    theme: Theme = Field(
        "dark",
        title=trans._("Theme"),
        description=trans._("Select the user interface theme."),
        env="napari_theme",
    )
    highlight_thickness: HighlightThickness = Field(  # type: ignore [valid-type]
        1,
        title=trans._("Highlight thickness"),
        description=trans._(
            "Select the highlight thickness when hovering over shapes/points."
        ),
    )
    layer_tooltip_visibility: bool = Field(
        False,
        title=trans._("Show layer tooltips"),
        description=trans._("Toggle to display a tooltip on mouse hover."),
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']

    def refresh_themes(self):
        """Updates theme data.
        This is not a fantastic solution but it works. Each time a new theme is
        added (either by a plugin or directly by the user) the enum is updated in
        place, ensuring that Preferences dialog can still be opened.
        """
        self.schema()["properties"]["theme"].update(enum=available_themes())
