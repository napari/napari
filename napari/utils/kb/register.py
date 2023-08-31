from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from app_model.expressions import Expr
from app_model.registries import KeyBindingsRegistry
from app_model.registries._keybindings_reg import _RegisteredKeyBinding
from app_model.types import KeyBinding, KeyBindingRule


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
    when: Optional[Expr] = field(compare=False)
    block_rule: bool = field(init=False)
    negate_rule: bool = field(init=False)

    def __post_init__(self):
        self.block_rule = self.command_id == ''
        self.negate_rule = self.command_id.startswith('-')


class NapariKeyBindingsRegistry(KeyBindingsRegistry):
    """Registry for key bindings.

    Attributes
    ----------
    keymap : Dict[int, List[KeyBindingEntry]]
        Keymap generated from registered key bindings.
    """

    def __init__(self) -> None:
        self.keymap: Dict[int, List[KeyBindingEntry]] = {}

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

        Returns
        -------
        Optional[Callable[[], None]]
            A callable that can be used to unregister the keybinding
        """
        if plat_keybinding := rule._bind_to_current_platform():
            entry = KeyBindingEntry(
                command_id=command_id,
                weight=rule.weight,
                when=rule.when,
            )

            print(entry)

            kb = KeyBinding.validate(plat_keybinding).to_int()
            if kb not in self.keymap:
                entries = []
                self.keymap[kb] = entries
            else:
                entries = self.keymap[kb]

            entries.append(entry)

            self.registered.emit()

            def _dispose() -> None:
                entries.remove(entry)
                if len(entries) == 0:
                    del self.keymap[kb]

            return _dispose
        return None  # pragma: no cover

    def __iter__(self) -> Iterator[Tuple[int, List[KeyBindingEntry]]]:
        yield from self.keymap.items()

    def __repr__(self) -> str:
        return repr(self.keymap)

    def get_keybinding(
        self, command_id: str
    ) -> Optional[_RegisteredKeyBinding]:
        for kb, entries in self.keymap.items():
            for entry in entries:
                if entry.command_id == command_id:
                    return _RegisteredKeyBinding(
                        keybinding=KeyBinding.from_int(kb),
                        command_id=command_id,
                        weight=entry.weight,
                        when=entry.when,
                    )
        return None
