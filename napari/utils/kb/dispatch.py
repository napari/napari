from threading import Timer
from typing import Dict, List, Mapping, Optional, Set

from app_model.expressions import Context
from app_model.types import KeyChord, KeyCode, KeyMod
from app_model.types._constants import OperatingSystem

from napari.utils.kb.constants import PRESS_HOLD_DELAY_MS, VALID_KEYS
from napari.utils.kb.register import KeyBindingEntry, NapariKeyBindingsRegistry
from napari.utils.kb.util import create_conflict_filter, key2mod


def find_active_match(
    entries: List[KeyBindingEntry], context: Mapping[str, object]
) -> Optional[KeyBindingEntry]:
    ignored_commands = []

    for entry in reversed(entries):
        if entry.when.eval(context):
            if entry.block_rule:
                return None

            if entry.negate_rule:
                ignored_commands.append(entry.command_id[1:])
            elif entry.command_id not in ignored_commands:
                return entry
    return None


def has_conflicts(
    key: int,
    keymap: Dict[int, List[KeyBindingEntry]],
    context: Mapping[str, object],
) -> bool:
    conflict_filter = create_conflict_filter(key)

    for _, entries in filter(conflict_filter, keymap.items()):
        if find_active_match(entries, context):
            return True

    return False


class KeyBindingDispatcher:
    def __init__(
        self, registry: NapariKeyBindingsRegistry, context: Context
    ) -> None:
        self.registry = registry
        self.context = context
        self.is_prefix: bool = False
        self.prefix: int = 0
        self.timer: Optional[Timer] = None
        self.active_combo: int = 0

        self._active_match_cache = {}
        self._conflicts_cache = {}

        self.context.changed.connect(self._on_context_change)

    def _on_context_change(self, changes: Set[str]):
        self._active_match_cache.clear()
        self._conflicts_cache.clear()

    def find_active_match(self, key: int) -> Optional[KeyBindingEntry]:
        try:
            return self._active_match_cache[key]
        except KeyError:
            entries = self.registry.keymap.get(key)
            if not entries:
                match = None
            else:
                match = find_active_match(entries, self.context)

            self._active_match_cache[key] = match

            return match

    def has_conflicts(self, key: int) -> bool:
        try:
            return self._conflicts_cache[key]
        except KeyError:
            conflicts = has_conflicts(key, self.registry.keymap, self.context)
            self._conflicts_cache[key] = conflicts
            return conflicts

    def on_key_press(self, mods: KeyMod, key: KeyCode):
        self.is_prefix = False
        self.active_combo = 0
        if self.timer:
            self.timer.cancel()
            self.timer = None
        if key not in VALID_KEYS:
            # ignore input
            self.prefix = 0
            return

        keymod = key2mod(key, OperatingSystem.current())

        if keymod is not None and not self.prefix:
            # single modifier dispatch only works on first part of key binding

            if mods & keymod:
                mods ^= keymod

            if mods == KeyMod.NONE and (
                match := self.find_active_match(keymod)
            ):
                # single modifier
                self.active_combo = key
                if self.has_conflicts(keymod):
                    # conflicts; exec after delay
                    self.timer = Timer(
                        PRESS_HOLD_DELAY_MS / 1000,
                        lambda: self.exec_press(match.command_id),
                    )
                    self.timer.start()
                else:
                    # no conflicts; exec immediately
                    self.exec_press(match.command_id)
        else:
            # non-modifier base key
            key_seq = mods | key
            if self.prefix:
                key_seq = KeyChord(self.prefix, key_seq)

            if match := self.find_active_match(key_seq):
                self.active_combo = mods | key
                if not self.prefix and self.has_conflicts(key_seq):
                    # first part of key binding, check for conflicts
                    self.is_prefix = True
                    return
                self.exec_press(match.command_id)

    def on_key_release(self, mods: KeyMod, key: KeyCode):
        if self.active_combo & key:
            if self.is_prefix:
                self.prefix = self.active_combo
                self.prefix = False
                return

            keymod = key2mod(key)

            if keymod is not None:
                # modifier base key
                if self.timer is not None:
                    # active timer, execute immediately
                    if not self.timer.finished.is_set():
                        # not already executed
                        self.timer.cancel()
                        self.exec_press(key)
                    self.timer = None
                    self.exec_release(key)
                    self.active_combo = 0
            else:
                # release segment of key binding
                self.exec_release(self.active_combo)

    def exec_press(self, key: int):
        print(f"press {self.find_active_match(key).command_id}")

    def exec_release(self, key: int):
        print(f"release {self.find_active_match(key).command_id}")
