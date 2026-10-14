from pydantic import AliasChoices, Field

from napari.settings._base import EventedSettings


class AdvancedSettings(EventedSettings):
    multisampling: bool = Field(
        False,
        title='Enable global multisampling.',
        description='Multisampling (antialiasing) improves quality by rendering at higher resolution to reduce aliasing, at the cost of some performance.',
        json_schema_extra={'requires_restart': True},
    )

    autoswap_buffers: bool = Field(
        False,
        title='Enable autoswapping rendering buffers.',
        description='Autoswapping rendering buffers improves quality by reducing tearing artifacts, while sacrificing some performance.',
        validation_alias=AliasChoices('autoswap_buffers', 'napari_autoswap'),
        json_schema_extra={'requires_restart': True},
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ('schema_version',)
