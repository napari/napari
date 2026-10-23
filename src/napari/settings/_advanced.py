from pydantic import Field

from napari.settings._base import EventedSettings


class AdvancedSettings(EventedSettings):
    multisampling: bool = Field(
        False,
        title='Enable global multisampling.',
        description='Multisampling (antialiasing) improves quality by rendering at higher resolution to reduce aliasing, at the cost of some performance.',
        json_schema_extra={'requires_restart': True},
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ('schema_version',)

    paint_fill_completion_radius: float = Field(
        default=1.5,
        title='Brush size multiplier within which Labels paint-and-fill autocompletes.',
        description='When painting and filling, the brush stroke will be completed and auto-filled\n'
        'once the mouse re-approaches the starting point within a radius equal\n'
        'to the brush size multiplied by this value.',
    )
