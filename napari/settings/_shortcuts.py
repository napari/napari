from typing import Dict, List

from pydantic import Field

from ..utils.events.evented_model import EventedModel
from ..utils.shortcuts import default_shortcuts
from ..utils.translations import trans


class ShortcutsSettings(EventedModel):
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
