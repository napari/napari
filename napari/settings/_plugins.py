from typing import Dict, List, Set
from enum import Enum

from pydantic import Field
from typing_extensions import TypedDict

from ..utils.events.evented_model import EventedModel
from ..utils.translations import trans
from ..utils.misc import running_as_bundled_app


class PluginHookOption(TypedDict):
    """Custom type specifying plugin, hook implementation function name, and enabled state."""

    plugin: str
    enabled: bool


CallOrderDict = Dict[str, List[PluginHookOption]]


class PluginAPI(str, Enum):
    napari_hub = 'napari Hub'
    pypi = 'PyPI'


class PluginsSettings(EventedModel):
    plugin_api: PluginAPI = Field(
        PluginAPI.napari_hub,
        title=trans._("Plugin API"),
        description=trans._(
            "Use the following API for querying plugin information.",
        ),
    )
    check_conda_forge: bool = Field(
        running_as_bundled_app(),
        title=trans._("Check conda-forge availability"),
        description=trans._(
            "Check if the plugin is available for install from conda-forge."
        ),
    )
    call_order: CallOrderDict = Field(
        default_factory=dict,
        title=trans._("Plugin sort order"),
        description=trans._(
            "Sort plugins for each action in the order to be called.",
        ),
    )
    disabled_plugins: Set[str] = Field(
        set(),
        title=trans._("Disabled plugins"),
        description=trans._(
            "Plugins to disable on application start.",
        ),
    )
    extension2reader: Dict[str, str] = Field(
        default_factory=dict,
        title=trans._('File extension readers'),
        description=trans._(
            'Assign file extensions to specific reader plugins'
        ),
    )
    extension2writer: Dict[str, str] = Field(
        default_factory=dict,
        title=trans._('Writer plugin extension association.'),
        description=trans._(
            'Assign file extensions to specific writer plugins'
        ),
    )

    class Config:
        use_enum_values = False

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = [
            'schema_version',
            'disabled_plugins',
            'extension2writer',
        ]

        # if running_as_bundled_app():
        #     preferences_exclude.append('plugin_api')
        #     preferences_exclude.append('check_conda_forge')
