import logging
from typing import Dict, List, Mapping, Optional, Set

from app_model.expressions import Context
from app_model.types import KeyChord, KeyCode, KeyMod
from app_model.types._constants import OperatingSystem
from psygnal import Signal

from napari._app_model.key_bindings.constants import VALID_KEYS, DispatchFlags
from napari._app_model.key_bindings.register import (
    KeyBindingEntry,
    NapariKeyBindingsRegistry,
)
from napari._app_model.key_bindings.util import create_conflict_filter, key2mod

logger = logging.getLogger(__name__)


def next_active_match(
    entries: List[KeyBindingEntry], context: Mapping[str, object]
) -> Optional[KeyBindingEntry]:
    """Find and yield active matches while traversing through the entries.

    See `NAP 7 <https://napari.org/dev/naps/7-key-binding-dispatch.html#key-binding-properties>`_.
    for more information.

    Parameters
    ----------
    entries: List[KeyBindingEntry]
        Pre-sorted entries to search through.
    context: Mapping[str, object]
        Context with which to evaluate the entries.

    Yields
    ------
    match: KeyBindingEntry or None
        Next match encountered, if found.
    """
    ignored_commands = []

    for entry in reversed(entries):
        if entry.when is None or entry.when.eval(context):
            if entry.block_rule:
                yield None
                break
            elif entry.negate_rule:
                command_id = entry.command_id[1:]
                ignored_commands.append(command_id)
            elif entry.command_id not in ignored_commands:
                yield entry


def find_active_match(
    entries: List[KeyBindingEntry], context: Mapping[str, object]
) -> Optional[KeyBindingEntry]:
    """Find and return the first active match.

    See `NAP 7 <https://napari.org/dev/naps/7-key-binding-dispatch.html#key-binding-properties>`_.
    for more information.

    Parameters
    ----------
    entries: List[KeyBindingEntry]
        Pre-sorted entries to search through.
    context: Mapping[str, object]
        Context with which to evaluate the entries.

    Returns
    -------
    match: KeyBindingEntry or None
        First match encountered, if found.
    """
    return next(next_active_match(entries, context), None)


def has_conflicts(
    key: int,
    keymap: Dict[int, List[KeyBindingEntry]],
    context: Mapping[str, object],
) -> bool:
    """Check if the given key has a conflict. Only works for the first part of a key binding.

    See `NAP 7 <https://napari.org/dev/naps/7-key-binding-dispatch.html#key-binding-properties>`_.
    for more information.

    Parameters
    ----------
    key: int
        Key, modifier, or key combo to check.
    keymap: Dict[int, List[KeyBindingEntry]]
        Keymap to check for conflicts.
    context: Mapping[str, object]
        Context with which to evaluate the entries.

    Returns
    -------
    bool
        If the given key has other active conflicts.
    """
    conflict_filter = create_conflict_filter(key)

    for _, entries in filter(
        lambda kv: conflict_filter(kv[0]), keymap.items()
    ):
        if find_active_match(entries, context):
            return True

    return False


class KeyBindingDispatcher:
    """Dispatcher for key binding system. Does not execute commands itself,
    instead it relays information using the `dispatch` signal.

    Parameters
    ----------
    registry: NapariKeyBindingsRegistry
        Registry providing the keymap to use for resolving key bindings.
    context: Context
        Mutable evented context with which to evaluate the key binding entries.
    os: Optional[OperatingSystem]
        Operating system with which to translate key bindings.

    Attributes
    ----------
    dispatch: Signal(DispatchFlags, Optional[str])
        Signal used to dispatch key binding related logic, containing flags for
        how the dispatch should be done as well as the command, if found.
    active_combo: int
        Currently active key combo.
    active_command: Optional[str]
        Current command being processed.
    is_prefix: bool
        Whether the current active combo is a prefix for another key binding.
    prefix: int
        Current active prefix.
    """

    dispatch = Signal(DispatchFlags, Optional[str])

    def __init__(
        self,
        registry: NapariKeyBindingsRegistry,
        context: Context,
        os: Optional[OperatingSystem] = None,
    ) -> None:
        self.registry = registry
        self.context = context
        if os is None:
            os = OperatingSystem.current()
        self.os = os

        self.active_combo: int = 0
        self.active_command: Optional[str] = None
        self.is_prefix: bool = False
        self.prefix: int = 0

        self._active_match_cache = {}
        self._conflicts_cache = {}
        self._active_keymap = None

        self.context.changed.connect(self._on_context_change)

    def _on_context_change(self, changes: Set[str]):
        logger.info('clearing cache: context change detected')
        logger.debug('current context: %s', self.context)
        self._active_match_cache.clear()
        self._conflicts_cache.clear()
        self._active_keymap = None

    def find_active_match(self, key: int) -> Optional[KeyBindingEntry]:
        """Find the active match for the key sequence provided.

        Parameters
        ----------
        key: int
            Key sequence to search.

        Returns
        -------
        match: Optional[KeyBindingEntry]
            Match, if found.
        """
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
        """Find if the given key sequence has a conflict. Only works for the first part of a key binding.

        Parameters
        ----------
        key: int
            Key or key combo to check.

        Returns
        -------
        bool
            Whether conflicts were found.
        """
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
        """Processes a key press.

        See `NAP 7 <https://napari.org/dev/naps/7-key-binding-dispatch.html#key-binding-properties>`_.
        for more information.

        Parameters
        ----------
        mods: KeyMod
            Modifiers held during the press.
        key: KeyCode
            Base key that was pressed.
        """
        logger.info('key press %s with mods %s', key, mods)
        logger.debug(
            'active combo: %s, prefix: %s', self.active_combo, self.prefix
        )
        self.is_prefix = False
        self.active_combo = 0
        flags = DispatchFlags.RESET
        command_id = None

        keymod = key2mod(key, self.os)

        if keymod and not self.prefix:
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
        elif key in VALID_KEYS:
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
        else:
            # ignore input
            self.prefix = 0

        self.active_command = command_id
        self.dispatch(flags, command_id)
        logger.info('dispatching %s with flags %s', command_id, flags)

    def on_key_release(self, mods: KeyMod, key: KeyCode):
        """Processes a key release.

        See `NAP 7 <https://napari.org/dev/naps/7-key-binding-dispatch.html#key-binding-properties>`_.
        for more information.

        Parameters
        ----------
        mods: KeyMod
            Modifiers held during the release.
        key: KeyCode
            Base key that was released.
        """
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

    @property
    def active_keymap(self) -> Mapping[int, str]:
        """Mapping[int, str]: Active mapping of keys to commands given the current context."""
        if self._active_keymap is None:
            self._active_keymap = {}
            for key in self.registry.keymap:
                if match := self.find_active_match(key):
                    self._active_keymap[key] = match
        return self._active_keymap
