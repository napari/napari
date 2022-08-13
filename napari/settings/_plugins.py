from typing import Dict, List, Set

from pydantic import Field
from typing_extensions import TypedDict

from ..utils.misc import running_as_bundled_app, running_as_constructor_app
from ..utils.translations import trans
from ._base import EventedSettings


class PluginHookOption(TypedDict):
    """Custom type specifying plugin, hook implementation function name, and enabled state."""

    plugin: str
    enabled: bool


CallOrderDict = Dict[str, List[PluginHookOption]]


class PluginsSettings(EventedSettings):
    use_npe2_adaptor: bool = Field(
        False,
        title=trans._("Use npe2 adaptor"),
        description=trans._(
            "Use npe2-adaptor for first generation plugins.\nWhen an npe1 plugin is found, this option will\nimport its contributions and create/cache\na 'shim' npe2 manifest that allows it to be treated\nlike an npe2 plugin (with delayed imports, etc...)",
        ),
        requires_restart=True,
    )
    plugin_api: str = Field(
        'pypi',
        title=trans._("Plugin API"),
        description=trans._("(Deprecated)."),
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

        if running_as_bundled_app() or running_as_constructor_app():
            preferences_exclude.append('plugin_api')
