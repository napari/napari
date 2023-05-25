from typing import Dict, List

from pydantic import Field, validator

from napari.utils.events.evented_model import EventedModel
from napari.utils.key_bindings import KeyBinding, coerce_keybinding
from napari.utils.shortcuts import default_shortcuts
from napari.utils.translations import trans


class ShortcutsSettings(EventedModel):
    shortcuts: Dict[str, List[KeyBinding]] = Field(
        default_shortcuts,
        title=trans._("shortcuts"),
        description=trans._(
            "Set keyboard shortcuts for actions.",
        ),
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']

    @validator('shortcuts', allow_reuse=True)
    def shortcut_validate(cls, v):
        for name, value in default_shortcuts.items():
            if name not in v:
                v[name] = value

        return {
            name: [coerce_keybinding(kb) for kb in value]
            for name, value in v.items()
        }
