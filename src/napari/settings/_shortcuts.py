from __future__ import annotations

from napari._pydantic_compat import Field, validator
from napari.utils.events.evented_model import EventedModel
from napari.utils.key_bindings import KeyBinding, coerce_keybinding
from napari.utils.shortcuts import default_shortcuts
from napari.utils.translations import trans


def get_app_default_shortcuts() -> dict[str, dict[str, list[KeyBinding]]]:
    from napari._app_model import get_app_model

    return get_app_model().get_default_shortcuts()


class ShortcutsSettings(EventedModel):
    shortcuts: dict[str, list[KeyBinding]] = Field(
        default_shortcuts,
        title=trans._('shortcuts'),
        description=trans._(
            'Set keyboard shortcuts for actions.',
        ),
    )
    app_shortcuts: dict[str, dict[str, list[KeyBinding]]] = Field(
        default_factory=get_app_default_shortcuts,
        title=trans._('app shortcuts'),
        description=trans._(
            'Set keyboard shortcuts for application-level actions.'
        ),
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ('schema_version',)

    @validator('shortcuts', allow_reuse=True, pre=True)
    def shortcut_validate(
        cls, v: dict[str, list[KeyBinding | str]]
    ) -> dict[str, list[KeyBinding]]:
        for name, value in default_shortcuts.items():
            if name not in v:
                v[name] = value

        return {
            name: [coerce_keybinding(kb) for kb in value]
            for name, value in v.items()
        }
