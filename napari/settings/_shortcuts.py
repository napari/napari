from typing import List, Optional

from app_model.types import KeyBinding

from napari._pydantic_compat import BaseModel, Field, validator
from napari.utils.action_manager import new_name_to_old
from napari.utils.events.evented_model import EventedModel
from napari.utils.key_bindings.legacy import coerce_keybinding
from napari.utils.key_bindings.util import kb2str, validate_key_binding
from napari.utils.translations import trans


class ShortcutRule(BaseModel):
    key: str = Field(allow_mutation=False)
    command: str = Field(allow_mutation=False)
    when: Optional[str] = Field(None, allow_mutation=False)

    class Config:
        validate_assignment = True


class ShortcutsSettings(EventedModel):
    shortcuts: List[ShortcutRule] = Field(
        default=[],
        title=trans._("shortcuts"),
        description=trans._(
            "Set keyboard shortcuts for actions.",
        ),
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ('schema_version',)

    def remove_shortcut(self, key, command, when=None):
        add_rule = ShortcutRule(key=key, command=command, when=when)
        negate_rule = ShortcutRule(key=key, command=f'-{command}', when=when)

        if negate_rule not in self.shortcuts:
            if add_rule in self.shortcuts:
                self.shortcuts.remove(add_rule)
            else:
                self.shortcuts.append(negate_rule)
            self.events.shortcuts()

    def add_shortcut(self, key, command, when=None):
        add_rule = ShortcutRule(key=key, command=command, when=when)
        negate_rule = ShortcutRule(key=key, command=f'-{command}', when=when)

        changed = False

        if negate_rule in self.shortcuts:
            self.shortcuts.remove(negate_rule)
            changed = True

        if add_rule not in self.shortcuts:
            self.shortcuts.append(add_rule)
            changed = True

        if changed:
            self.events.shortcuts()

    def overwrite_shortcut(self, old_key, new_key, command, when=None):
        with self.events.shortcuts.blocker():
            self.remove_shortcut(old_key, command, when)
            self.add_shortcut(new_key, command, when)
        self.events.shortcuts()

    @validator('shortcuts', allow_reuse=True, pre=True)
    def shortcut_validate(cls, shortcuts):
        if isinstance(shortcuts, dict):
            # legacy case
            from napari.constants import DEFAULT_SHORTCUTS

            new = []
            skip = []

            # handle defaults
            for command, entries in DEFAULT_SHORTCUTS.items():
                # special casing since paste_shape was renamed to paste_shapes
                if command == 'napari:shapes:paste_shapes':
                    command = command[:-1]
                legacy_name = new_name_to_old(command)

                if legacy_entries := shortcuts.get(legacy_name):
                    skip.append(legacy_name)
                    default_entries = [
                        KeyBinding.validate(entry._bind_to_current_platform())
                        for entry in entries
                    ]
                    legacy_kbs = [
                        validate_key_binding(
                            coerce_keybinding(entry), warn=False
                        )
                        for entry in legacy_entries
                    ]

                    for i, kb in enumerate(legacy_kbs):
                        if kb in default_entries:
                            # redundant
                            continue

                        when = None

                        prefix, group, *suffix = command.split(':')
                        if group in (
                            'image',
                            'labels',
                            'points',
                            'shapes',
                            'surface',
                            'tracks',
                            'vectors',
                        ):
                            when = f'active_layer_type == "{group}"'

                        if (
                            i < len(default_entries)
                            and default_entries[i] not in legacy_kbs
                        ):
                            # unbind default
                            new.append(
                                ShortcutRule(
                                    key=kb2str(default_entries[i]),
                                    command=f'-{command}',
                                    when=when,
                                )
                            )

                        new.append(
                            ShortcutRule(
                                key=kb2str(kb), command=command, when=when
                            )
                        )

            for command, entries in shortcuts.items():
                if command in skip:
                    continue
                for entry in entries:
                    new.append(
                        ShortcutRule(
                            key=str(kb2str(entry)),
                            command=command,
                        )
                    )

            return new

        return shortcuts
