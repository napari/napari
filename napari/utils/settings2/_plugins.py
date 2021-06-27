from typing import Dict, List, Set, Tuple, TypedDict, Union

from pydantic import Field

from ..events.evented_model import EventedModel
from ..translations import trans
from ._fields import SchemaVersion


class PluginHookOption(TypedDict):
    """Custom type specifying plugin and enabled state."""

    plugin: str
    enabled: bool


CallOrderDict = Dict[str, List[PluginHookOption]]


class PluginsSettings(EventedModel):
    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option
    schema_version: Union[SchemaVersion, Tuple[int, int, int]] = (0, 1, 1)
    call_order: CallOrderDict = Field(
        None,
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

    class Config:
        # Pydantic specific configuration
        schema_extra = {
            "title": trans._("Plugins"),
            "description": trans._("Plugins settings."),
            "section": "plugins",
        }

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version', 'disabled_plugins']
