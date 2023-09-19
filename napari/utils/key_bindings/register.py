import warnings
from bisect import insort_left
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from app_model.expressions import Expr
from app_model.registries import KeyBindingsRegistry
from app_model.registries._keybindings_reg import _RegisteredKeyBinding
from app_model.types import KeyBinding, KeyBindingRule
from psygnal import Signal

from napari.utils.key_bindings.util import validate_key_binding


@dataclass(order=True)
class KeyBindingEntry:
    """Internal entry for a key binding.

    See `NAP 7 <https://napari.org/dev/naps/7-key-binding-dispatch.html#key-binding-properties>`_.
    for more information.

    Parameters
    ----------
    command_id : str
        The unique identifier of the command that will be executed by this key binding.
    weight : int
        The main determinant of key binding priority. A higher value means a higher priority.
    when : Optional[Expr]
        The context expression that is evaluated to determine whether the rule is active;
        if not provided, the rule is always considered active.

    Attributes
    ----------
    block_rule : bool
        Enabled if `command_id == ''` and disables all key bindings of their weight and below.
    negate_rule : bool
        Enabled if `command_id` is prefixed with `-` and disables all key bindings of their weight
        and below with the same sequence bound to this command.
    """

    command_id: str = field(compare=False)
    weight: int
    when: Optional[Expr] = field(compare=False, default=None)
    _index: int = field(compare=False, default=0)
    block_rule: bool = field(init=False)
    negate_rule: bool = field(init=False)

    def __post_init__(self):
        self.block_rule = self.command_id == ''
        self.negate_rule = self.command_id.startswith('-')


def filter_entries_by_command(
    entries: List[KeyBindingEntry], command_id: str
) -> Iterator[KeyBindingEntry]:
    """Filter entries to ones directly referencing the specified command (includes negate rules).

    Parameters
    ----------
    entries : List[KeyBindingEntry]
        Entries to filter.
    command_id : str
        Command to filter for.

    Returns
    -------
    Iterator[KeyBindingEntry]
        Filtered entries.
    """
    return (
        entry
        for entry in entries
        if entry.command_id[entry.negate_rule :] == command_id
    )


def group_entries_by_when(
    entries: Iterable[KeyBindingEntry],
) -> Dict[Optional[str], List[KeyBindingEntry]]:
    """Group entries by their when condition.

    Parameters
    ----------
    entries : Iterable[KeyBindingEntry]
        Entries to group.

    Returns
    -------
    groups : Dict[Optional[str], List[KeyBindingEntry]]
        Grouped entries.
    """
    # hashing isn't consistent with expressions; use a str instead
    groups: Dict[Optional[str], List[KeyBindingEntry]] = {}

    for entry in entries:
        when = None
        if entry.when is not None:
            when = str(entry.when)

        if when in groups:
            groups[when].append(entry)
        else:
            groups[when] = [entry]

    return groups


class NapariKeyBindingsRegistry(KeyBindingsRegistry):
    """Registry for key bindings.

    Attributes
    ----------
    keymap : Dict[int, List[KeyBindingEntry]]
        Keymap generated from registered key bindings.
    """

    unregistered = Signal()

    def __init__(self) -> None:
        self.keymap: Dict[int, List[KeyBindingEntry]] = {}
        self._index = 0

    def register_keybinding_rule(
        self, command_id: str, rule: KeyBindingRule
    ) -> Optional[Callable[[], None]]:
        """Register a new keybinding rule.

         Parameters
        ----------
        command_id : str
            Command identifier that should be run when the keybinding is triggered
        rule : KeyBindingRule
            KeyBinding information

        Raises
        ------
        TypeError
            When the key binding is invalid

        Warns
        -----
        UserWarning
            When the key binding is a single modifier encoded as a KeyCode instead of a KeyMod

        Returns
        -------
        Optional[Callable[[], None]]
            A callable that can be used to unregister the keybinding
        """
        if plat_keybinding := rule._bind_to_current_platform():
            entry = KeyBindingEntry(
                command_id=command_id,
                weight=rule.weight,
                _index=self._index,
                when=rule.when,
            )
            self._index += 1

            with warnings.catch_warnings(record=True) as w:
                key_bind = validate_key_binding(
                    KeyBinding.validate(plat_keybinding)
                )

            if len(w) == 1:
                warnings.warn(
                    f'{w[0].message} for entry {entry}',
                    UserWarning,
                    stacklevel=2,
                )

            kb = key_bind.to_int()

            if kb not in self.keymap:
                entries: List[KeyBindingEntry] = []
                self.keymap[kb] = entries
            else:
                entries = self.keymap[kb]

            insort_left(entries, entry)

            self.registered.emit()

            def _dispose() -> None:
                entries.remove(entry)
                self.unregistered.emit()
                if len(entries) == 0:
                    del self.keymap[kb]

            return _dispose
        return None  # pragma: no cover

    def __iter__(self) -> Iterator[Tuple[int, List[KeyBindingEntry]]]:
        yield from self.keymap.items()

    def __repr__(self) -> str:
        return repr(self.keymap)

    def discard_entries(self, weight_threshold: int) -> bool:
        """Discard all entries of the given weight threshold or higher.

        Parameters
        ----------
        weight_threshold : int
            Threshold for which to discard entries based on.

        Returns
        -------
        discarded : bool
            If any entries were discarded.
        """
        discarded = False
        del_keys = []

        for kb in self.keymap:
            entries = self.keymap[kb]
            culled_entries = [
                entry for entry in entries if entry.weight < weight_threshold
            ]
            if len(entries) != len(culled_entries):
                discarded = True
                self.keymap[kb] = culled_entries

            if len(culled_entries) == 0:
                del_keys.append(kb)

        for key in del_keys:
            del self.keymap[key]

        if discarded:
            self.unregistered()

        return discarded

    def get_keybinding(
        self, command_id: str
    ) -> Optional[_RegisteredKeyBinding]:
        return next(self.get_next_entry(command_id), None)

    def get_next_entry(
        self, command_id: str
    ) -> Iterator[_RegisteredKeyBinding]:
        for kb, entries in self.keymap.items():
            for entry in entries:
                if entry.command_id[entry.negate_rule :] == command_id:
                    yield _RegisteredKeyBinding(
                        keybinding=KeyBinding.from_int(kb),
                        command_id=entry.command_id,
                        weight=entry.weight,
                        when=entry.when,
                    )

    def get_non_canceling_entries(
        self, command_id: str
    ) -> List[_RegisteredKeyBinding]:
        """Get all entries for the given command that don't cancel each other out.

        Parameters
        ----------
        command_id : str
            Command to search for.

        Returns
        -------
        List[_RegisteredKeyBinding]
            Non canceling entries.
        """
        nc_entries: List[Tuple[KeyBindingEntry, _RegisteredKeyBinding]] = []

        for key, entries in self.keymap.items():
            kb = KeyBinding.from_int(key)
            groups = group_entries_by_when(
                filter_entries_by_command(entries, command_id)
            )

            for group_entries in groups.values():
                temp_entries: List[
                    Tuple[KeyBindingEntry, _RegisteredKeyBinding]
                ] = []

                for entry in group_entries:
                    if entry.block_rule or entry.negate_rule:
                        temp_entries.clear()
                    else:
                        temp_entries.append(
                            (
                                entry,
                                _RegisteredKeyBinding(
                                    keybinding=kb,
                                    command_id=command_id,
                                    weight=entry.weight,
                                    when=entry.when,
                                ),
                            )
                        )

                nc_entries.extend(temp_entries)

        return [
            e2 for (e1, e2) in sorted(nc_entries, key=lambda ab: ab[0]._index)
        ]
