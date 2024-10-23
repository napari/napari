from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from app_model.types import CommandRule

if TYPE_CHECKING:
    from qtpy import QtWidgets as QtW

    from napari._qt.widgets.qt_command_palette import QCommandPalette

    _WVDict = weakref.WeakValueDictionary[int, QtW.QWidget]


class CommandPalette:
    """The command palette interface."""

    def __init__(self, name: str) -> None:
        self._command_rules: list[CommandRule] = []
        self._parent_to_palette_map: dict[int, QCommandPalette] = {}
        self._palette_to_parent_map: _WVDict = weakref.WeakValueDictionary()
        self._name = name

    @property
    def command_rules(self) -> list[CommandRule]:
        """List of all the commands."""
        return self._command_rules.copy()

    def register(self, cmd: CommandRule) -> None:
        """Register a command to the palette."""
        # update defaults
        self._command_rules.append(cmd)
        return


_GLOBAL_PALETTES: dict[str, CommandPalette] = {}


def get_palette(name) -> CommandPalette:
    """
    Get the global command palette object.

    Examples
    --------
    >>> palette = get_palette()  # get the default palette
    >>> palette = get_palette("my_module")  # get a palette for specific app
    """
    global _GLOBAL_PALETTES

    if not isinstance(name, str):
        raise TypeError(f'Expected str, got {type(name).__name__}')
    if (palette := _GLOBAL_PALETTES.get(name)) is None:
        palette = _GLOBAL_PALETTES[name] = CommandPalette(name=name)
    return palette
