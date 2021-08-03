from typing import Dict, List, Tuple, Union

from pydantic import Field

from ..utils.events.evented_model import EventedModel
from ..utils.shortcuts import default_shortcuts
from ..utils.translations import trans
from ._fields import SchemaVersion


class ShortcutsSettings(EventedModel):
    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option
    schema_version: Union[SchemaVersion, Tuple[int, int, int]] = (0, 1, 1)
    shortcuts: Dict[str, List[str]] = Field(
        default_shortcuts,
        title=trans._("shortcuts"),
        description=trans._(
            "Set keyboard shortcuts for actions.",
        ),
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']
