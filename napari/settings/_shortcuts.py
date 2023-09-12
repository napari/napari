from typing import List, Optional

from app_model.types import KeyBinding
from pydantic import Field, validator

from napari.utils.action_manager import new_name_to_old
from napari.utils.events.evented_model import EventedModel
from napari.utils.key_bindings.legacy import coerce_keybinding
from napari.utils.translations import trans


class ShortcutRule(EventedModel):
    key: str
    command: str
    when: Optional[str] = None


class ShortcutsSettings(EventedModel):
    shortcuts: List[ShortcutRule] = Field(
        default_factory=list,
        title=trans._("shortcuts"),
        description=trans._(
            "Set keyboard shortcuts for actions.",
        ),
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ('schema_version',)

    def remove_shortcut(self, key, command, when=None):
        rule = ShortcutRule(key=key, command=f'-{command}', when=when)

        if rule not in self.shortcuts:
            self.shortcuts.append(rule)

    def add_shortcut(self, key, command, when=None):
        rule = ShortcutRule(key=key, command=command, when=when)

        if rule not in self.shortcuts:
            self.shortcuts.append(rule)

    def overwrite_shortcut(self, new_key, old_key, command, when=None):
        self.remove_shortcut(old_key, command, when)
        self.add_shortcut(new_key, command, when)

    @validator('shortcuts', allow_reuse=True)
    def shortcut_validate(cls, shortcuts):
        if isinstance(shortcuts, dict):
            # legacy case
            from napari._app_model.constants import DEFAULT_SHORTCUTS

            new = []
            skip = []

            # handle defaults
            for command, entries in DEFAULT_SHORTCUTS.items():
                legacy_name = new_name_to_old(command)

                if legacy_entries := shortcuts.get(legacy_name):
                    skip.append(legacy_name)
                    default_entries = [
                        KeyBinding.validate(entry.primary) for entry in entries
                    ]

                    for i, entry in enumerate(legacy_entries):
                        kb = coerce_keybinding(entry)
                        if kb in default_entries:
                            # redundant
                            continue

                        if i < len(default_entries):
                            # unbind default
                            new.append(
                                ShortcutRule(
                                    key=str(default_entries[i]).lower(),
                                    command=f'-{command}',
                                )
                            )

                        new.append(
                            ShortcutRule(key=str(kb).lower(), command=command)
                        )

            for command, entries in shortcuts.items():
                if command in skip:
                    continue
                for entry in entries:
                    new.append(
                        ShortcutRule(
                            key=str(coerce_keybinding(entry)).lower(),
                            command=command,
                        )
                    )

            return new

        return shortcuts
