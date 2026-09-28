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
