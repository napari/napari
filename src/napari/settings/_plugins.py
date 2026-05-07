from pydantic import Field
from pydantic_settings import SettingsConfigDict
from typing_extensions import TypedDict

from napari.settings._base import EventedSettings


class PluginHookOption(TypedDict):
    """Custom type specifying plugin, hook implementation function name, and enabled state."""

    plugin: str
    enabled: bool


CallOrderDict = dict[str, list[PluginHookOption]]


class PluginsSettings(EventedSettings):
    disabled_plugins: set[str] = Field(
        set(),
        title='Disabled plugins',
        description='Plugins to disable on application start.',
    )
    extension2reader: dict[str, str] = Field(
        default_factory=dict,
        title='File extension readers',
        description='Assign file extensions to specific reader plugins',
    )
    extension2writer: dict[str, str] = Field(
        default_factory=dict,
        title='Writer plugin extension association.',
        description='Assign file extensions to specific writer plugins',
    )

    model_config = SettingsConfigDict(use_enum_values=False)

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = (
            'schema_version',
            'disabled_plugins',
            'extension2writer',
        )
