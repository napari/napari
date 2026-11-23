from __future__ import annotations

from pydantic import Field, field_validator

from napari.utils.events.evented_model import EventedModel
from napari.utils.key_bindings import KeyBinding, coerce_keybinding
from napari.utils.shortcuts import _normalize_action_name, default_shortcuts


class ShortcutsSettings(EventedModel):
    shortcuts: dict[str, list[KeyBinding]] = Field(
        default_shortcuts,
        title='',
        description='Set keyboard shortcuts for actions.',
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ('schema_version',)

    @field_validator('shortcuts', mode='before')
    @classmethod
    def shortcut_validate(
        cls, v: dict[str, list[KeyBinding | str]]
    ) -> dict[str, list[KeyBinding]]:
        # Prefer an explicit dotted ID if both forms are present.
        v = {
            _normalize_action_name(name): value
            for name, value in v.items()
            if ':' not in name or _normalize_action_name(name) not in v
        }
        for name, value in default_shortcuts.items():
            if name not in v:
                # make a copy of the default value
                v[name] = list(value)

        return {
            name: [coerce_keybinding(kb) for kb in value]
            for name, value in v.items()
        }
