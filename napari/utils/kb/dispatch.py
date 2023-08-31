import logging
from typing import Dict, List, Mapping, Optional, Set

from app_model.expressions import Context
from app_model.types import KeyChord, KeyCode, KeyMod
from app_model.types._constants import OperatingSystem
from psygnal import Signal

from napari.utils.kb.constants import (
    VALID_KEYS,
    DispatchFlags,
)
from napari.utils.kb.register import KeyBindingEntry, NapariKeyBindingsRegistry
from napari.utils.kb.util import create_conflict_filter, key2mod

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def find_active_match(
    entries: List[KeyBindingEntry], context: Mapping[str, object]
) -> Optional[KeyBindingEntry]:
    ignored_commands = []

    for entry in reversed(entries):
        logger.debug('ENTER %s', entry)
        if entry.when is None or entry.when.eval(context):
            logger.debug('entry %s active', entry)
            if entry.block_rule:
                logger.debug('entry is block rule')
                return None

            if entry.negate_rule:
                command_id = entry.command_id[1:]
                logger.debug('entry negates %s', command_id)
                ignored_commands.append(entry.command_id)
            elif entry.command_id not in ignored_commands:
                logger.debug('entry ignored')
                return entry
    return None


def has_conflicts(
    key: int,
    keymap: Dict[int, List[KeyBindingEntry]],
    context: Mapping[str, object],
) -> bool:
    conflict_filter = create_conflict_filter(key)

    for _, entries in filter(
        lambda kv: conflict_filter(kv[0]), keymap.items()
    ):
        if find_active_match(entries, context):
            return True

    return False


class KeyBindingDispatcher:
    dispatch = Signal(DispatchFlags, Optional[str])

    def __init__(
        self,
        registry: NapariKeyBindingsRegistry,
        context: Context,
        os: Optional[OperatingSystem] = None,
    ) -> None:
        self.registry = registry
        self.context = context
        self.is_prefix: bool = False
        self.prefix: int = 0
        self.active_combo: int = 0
        self.active_command: Optional[str] = None

        if os is None:
            os = OperatingSystem.current()
        self.os = os

        self._active_match_cache = {}
        self._conflicts_cache = {}

        self.context.changed.connect(self._on_context_change)

    def _on_context_change(self, changes: Set[str]):
        logger.info('clearing cache: context change detected')
        logger.debug('current context: %s', self.context)
        self._active_match_cache.clear()
        self._conflicts_cache.clear()

    def find_active_match(self, key: int) -> Optional[KeyBindingEntry]:
        try:
            match = self._active_match_cache[key]
            logger.info('cached match found for %s: %s', key, match)
        except KeyError:
            logger.debug('cached match not found for %s', key)
            entries = self.registry.keymap.get(key)
            if not entries:
                match = None
            else:
                match = find_active_match(entries, self.context)

            self._active_match_cache[key] = match
            logger.info('saved match cache for %s: %s', key, match)

        return match

    def has_conflicts(self, key: int) -> bool:
        try:
            conflicts = self._conflicts_cache[key]
            logger.info('cached conflicts found for %s: %s', key, conflicts)
        except KeyError:
            logger.debug('cached conflicts not found for %s', key)
            conflicts = has_conflicts(key, self.registry.keymap, self.context)
            self._conflicts_cache[key] = conflicts
            logger.debug('saved conflict cache for %s: %s', key, conflicts)
        return conflicts

    def on_key_press(self, mods: KeyMod, key: KeyCode):
        logger.info('key press %s with mods %s', key, mods)
        logger.debug(
            'active combo: %s, prefix: %s', self.active_combo, self.prefix
        )
        self.is_prefix = False
        self.active_combo = 0
        flags = DispatchFlags.RESET
        command_id = None

        keymod = key2mod(key, self.os)

        if not keymod and key not in VALID_KEYS:
            # ignore input
            self.prefix = 0
        elif keymod and not self.prefix:
            # single modifier dispatch only works on first part of key binding
            flags |= DispatchFlags.SINGLE_MOD

            if mods & keymod:
                mods ^= keymod

            if mods == KeyMod.NONE and (
                match := self.find_active_match(keymod)
            ):
                # single modifier
                self.active_combo = key
                flags |= DispatchFlags.SINGLE_MOD
                if self.has_conflicts(keymod):
                    # conflicts; exec after delay
                    flags |= DispatchFlags.DELAY
                    command_id = match.command_id
        else:
            # non-modifier base key
            key_seq = mods | key
            self.active_combo = key_seq

            if self.prefix:
                flags |= DispatchFlags.TWO_PART
                key_seq = KeyChord(self.prefix, key_seq)

            if not self.prefix and self.has_conflicts(key_seq):
                # first part of key binding, check for conflicts
                self.is_prefix = True
            elif match := self.find_active_match(key_seq):
                command_id = match.command_id

        self.active_command = command_id
        self.dispatch(flags, command_id)
        logger.info('dispatching %s with flags %s', command_id, flags)

    def on_key_release(self, mods: KeyMod, key: KeyCode):
        logger.info('key release %s with mods %s', key, mods)
        logger.debug(
            'active combo: %s, prefix: %s', self.active_combo, self.prefix
        )
        if self.active_combo & key:
            flags = DispatchFlags.ON_RELEASE

            if self.prefix:
                flags |= DispatchFlags.TWO_PART
                self.prefix = 0

            if self.is_prefix:
                self.prefix = self.active_combo
                self.is_prefix = False
                self.active_combo = 0
            else:
                keymod = key2mod(key, os=self.os)

                if keymod is not None:
                    flags |= DispatchFlags.SINGLE_MOD

            self.dispatch(flags, self.active_command)
            logger.info(
                'dispatching %s with flags %s', self.active_command, flags
            )
            self.active_command = None
