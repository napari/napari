import sys
from typing import Union, cast

from pydantic import Field

from napari.settings._fields import Theme
from napari.utils.events.evented_model import ComparisonDelayer, EventedModel
from napari.utils.theme import available_themes, get_theme
from napari.utils.translations import trans


class AppearanceSettings(EventedModel):
    theme: Theme = Field(
        Theme("dark"),
        title=trans._("Theme"),
        description=trans._("Select the user interface theme."),
        env="napari_theme",
    )
    font_size: int = Field(
        12 if sys.platform == 'darwin' else 9,
        title=trans._("Font size"),
        description=trans._("Select the user interface font size."),
        ge=5,
        le=20,
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

    def update(
        self, values: Union['EventedModel', dict], recurse: bool = True
    ) -> None:
        if isinstance(values, self.__class__):
            values = values.dict()
        values = cast(dict, values)

        if "theme" in values and values["theme"] != self.theme:
            new_theme = get_theme(values["theme"])
            values["font_size"] = int(new_theme.font_size[:-2])
        super().update(values, recurse)

    def __setattr__(self, key, value):
        if key == "theme" and value != self.theme:
            with ComparisonDelayer(self):
                new_theme = get_theme(value)
                self.font_size = int(new_theme.font_size[:-2])
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ('schema_version',)

    def refresh_themes(self):
        """Updates theme data.
        This is not a fantastic solution but it works. Each time a new theme is
        added (either by a plugin or directly by the user) the enum is updated in
        place, ensuring that Preferences dialog can still be opened.
        """
        self.schema()["properties"]["theme"].update(enum=available_themes())
