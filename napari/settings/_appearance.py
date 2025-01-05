from typing import Union, cast

from napari._pydantic_compat import Field
from napari.settings._fields import Theme
from napari.utils.events.evented_model import ComparisonDelayer, EventedModel
from napari.utils.theme import available_themes, get_theme
from napari.utils.translations import trans


class HighlightSettings(EventedModel):
    highlight_thickness: int = Field(
        1,
        title=trans._('Highlight thickness'),
        description=trans._(
            'Select the highlight thickness when hovering over shapes/points.'
        ),
        ge=1,
        le=10,
    )
    highlight_color: list[float] = Field(
        [0.0, 0.6, 1.0, 1.0],
        title=trans._('Highlight color'),
        description=trans._(
            'Select the highlight color when hovering over shapes/points.'
        ),
    )


class AppearanceSettings(EventedModel):
    theme: Theme = Field(
        Theme('dark'),
        title=trans._('Theme'),
        description=trans._('Select the user interface theme.'),
        env='napari_theme',
    )
    font_size: int = Field(
        int(get_theme('dark').font_size[:-2]),
        title=trans._('Font size'),
        description=trans._('Select the user interface font size.'),
        ge=5,
        le=20,
    )
    highlight: HighlightSettings = Field(
        HighlightSettings(),
        title=trans._('Highlight'),
        description=trans._(
            'Select the highlight color and thickness to use when hovering over shapes/points.'
        ),
    )
    layer_tooltip_visibility: bool = Field(
        False,
        title=trans._('Show layer tooltips'),
        description=trans._('Toggle to display a tooltip on mouse hover.'),
    )
    update_status_based_on_layer: bool = Field(
        True,
        title=trans._('Update status based on layer'),
        description=trans._(
            'Calculate status bar based on current active layer and mouse position.'
        ),
    )

    def update(
        self, values: Union['EventedModel', dict], recurse: bool = True
    ) -> None:
        if isinstance(values, self.__class__):
            values = values.dict()
        values = cast(dict, values)

        # Check if a font_size change is needed when changing theme:
        # If the font_size setting doesn't correspond to the default value
        # of the current theme no change is done, otherwise
        # the font_size value is set to the new selected theme font size value
        if 'theme' in values and values['theme'] != self.theme:
            current_theme = get_theme(self.theme)
            new_theme = get_theme(values['theme'])
            if values['font_size'] == int(current_theme.font_size[:-2]):
                values['font_size'] = int(new_theme.font_size[:-2])
        super().update(values, recurse)

    def __setattr__(self, key: str, value: Theme) -> None:
        # Check if a font_size change is needed when changing theme:
        # If the font_size setting doesn't correspond to the default value
        # of the current theme no change is done, otherwise
        # the font_size value is set to the new selected theme font size value
        if key == 'theme' and value != self.theme:
            with ComparisonDelayer(self):
                new_theme = None
                current_theme = None
                if value in available_themes():
                    new_theme = get_theme(value)
                if self.theme in available_themes():
                    current_theme = get_theme(self.theme)
                if (
                    new_theme
                    and current_theme
                    and self.font_size == int(current_theme.font_size[:-2])
                ):
                    self.font_size = int(new_theme.font_size[:-2])
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ('schema_version',)

    def refresh_themes(self) -> None:
        """Updates theme data.
        This is not a fantastic solution but it works. Each time a new theme is
        added (either by a plugin or directly by the user) the enum is updated in
        place, ensuring that Preferences dialog can still be opened.
        """
        self.schema()['properties']['theme'].update(enum=available_themes())
